---
title: "手写Promise的相关方法"
date: 2019-11-23
permalink: /2019-11-23-promise-methods/
categories: ["源码精读", "Promise专题"]
---
## 摘要


Promise 作为 JS 社区的异步解决方案，为开发者提供了`.then()`、`Promise.resolve()`、`Promise.reject()`等基本方法。除此之外，为了更方便地组合和控制多个的 Promise 实例，也提供了`.all()`、`.race()`等方法。


本文会在 Promise 的基本方法上，手动实现更高级的方法，来加深对 Promise 的理解：

- 🤔️ 实现`Promise.all`
- 🤔️ 实现`Promise.race`
- 🤔️ 实现`Promise.any`
- 🤔️ 实现`Promise.allSettled`
- 🤔️ 实现`Promise.finally`

⚠️ 完整代码和用例请到[github.com/dongyuanxin/diy-promise](https://github.com/dongyuanxin/diy-promise)。


## 实现 Promise.all


### 过程


`Promise.all(iterators)`返回一个新的 Promise 实例。iterators 中包含外界传入的多个 promise 实例。


对于返回的新的 Promise 实例，有以下两种情况：

- 如果传入的所有 promise 实例的状态均变为`fulfilled`，那么返回的 promise 实例的状态就是`fulfilled`，并且其 value 是 传入的所有 promise 的 value 组成的数组。
- 如果有一个 promise 实例状态变为了`rejected`，那么返回的 promise 实例的状态立即变为`rejected`。

### 代码实现


实现思路：

- 传入的参数不一定是数组对象，可以是"遍历器"
- 传入的每个实例不一定是 promise，需要用`Promise.resolve()`包装
- 借助"计数器"，标记是否所有的实例状态均变为`fulfilled`

```typescript
Promise.myAll = function (iterators) {
    const promises = Array.from(iterators);
    const num = promises.length;
    const resolvedList = new Array(num);
    let resolvedNum = 0;

    return new Promise((resolve, reject) => {
        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then((value) => {
                    // 保存这个promise实例的value
                    resolvedList[index] = value;
                    // 通过计数器，标记是否所有实例均 fulfilled
                    if (++resolvedNum === num) {
                        resolve(resolvedList);
                    }
                })
                .catch(reject);
        });
    });
};
```


## 实现 Promise.race


### 过程


`Promise.race(iterators)`的传参和返回值与`Promise.all`相同。但其返回的 promise 的实例的状态和 value，完全取决于：传入的所有 promise 实例中，最先改变状态那个（不论是`fulfilled`还是`rejected`）。


### 代码实现


实现思路：

- 某传入实例`pending -> fulfilled`时，其 value 就是`Promise.race`返回的 promise 实例的 value
- 某传入实例`pending -> rejected`时，其 error 就是`Promise.race`返回的 promise 实例的 error

```typescript
Promise.myRace = function (iterators) {
    const promises = Array.from(iterators);

    return new Promise((resolve, reject) => {
        promises.forEach((promise, index) => {
            Promise.resolve(promise).then(resolve).catch(reject);
        });
    });
};
```


## 实现 Promise.any


### 过程


`Promise.any(iterators)`的传参和返回值与`Promise.all`相同。


如果传入的实例中，有任一实例变为`fulfilled`，那么它返回的 promise 实例状态立即变为`fulfilled`；如果所有实例均变为`rejected`，那么它返回的 promise 实例状态为`rejected`。


⚠️`Promise.all`与`Promise.any`的关系，类似于，`Array.prototype.every`和`Array.prototype.some`的关系。


### 代码实现


实现思路和`Promise.all`及其类似。不过由于对异步过程的处理逻辑不同，**因此这里的计数器用来标识是否所有的实例均 rejected**。


```typescript
Promise.any = function (iterators) {
    const promises = Array.from(iterators);
    const num = promises.length;
    const rejectedList = new Array(num);
    let rejectedNum = 0;

    return new Promise((resolve, reject) => {
        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then((value) => resolve(value))
                .catch((error) => {
                    rejectedList[index] = error;
                    if (++rejectedNum === num) {
                        reject(rejectedList);
                    }
                });
        });
    });
};

```


## 实现 Promise.allSettled


### 过程


`Promise.allSettled(iterators)`的传参和返回值与`Promise.all`相同。


根据[ES2020](https://github.com/tc39/proposal-promise-allSettled)，此返回的 promise 实例的状态只能是`fulfilled`。对于传入的所有 promise 实例，会等待每个 promise 实例结束，并且返回规定的数据格式。


如果传入 a、b 两个 promise 实例：a 变为 rejected，错误是 error1；b 变为 fulfilled，value 是 1。那么`Promise.allSettled`返回的 promise 实例的 value 就是：


```json
[
    { status: "rejected", value: error1 },
    { status: "fulfilled", value: 1 },
];
```


### 代码实现


实现中的计数器，用于统计所有传入的 promise 实例。


```typescript
const formatSettledResult = (success, value) =>
    success
        ? { status: "fulfilled", value }
        : { status: "rejected", reason: value };

Promise.allSettled = function (iterators) {
    const promises = Array.from(iterators);
    const num = promises.length;
    const settledList = new Array(num);
    let settledNum = 0;

    return new Promise((resolve) => {
        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then((value) => {
                    settledList[index] = formatSettledResult(true, value);
                    if (++settledNum === num) {
                        resolve(settledList);
                    }
                })
                .catch((error) => {
                    settledList[index] = formatSettledResult(false, error);
                    if (++settledNum === num) {
                        resolve(settledList);
                    }
                });
        });
    });
};
```


## Promise.all、Promise.any 和 Promise.allSettled 中计数器使用对比


这三个方法均使用了计数器来进行异步流程控制，下面表格横向对比不同方法中计数器的用途，来加强理解：


child_database


## 实现 Promise.prototype.finally


### 过程


它就是一个语法糖，在当前 promise 实例执行完 then 或者 catch 后，均会触发。


举个例子，一个 promise 在 then 和 catch 中均要打印时间戳：


```typescript
new Promise((resolve) => {
    setTimeout(() => resolve(1), 1000);
})
    .then((value) => console.log(Date.now()))
    .catch((error) => console.log(Date.now()));

```


现在这段一定执行的共同逻辑，就可以用`finally`简写为：


```typescript
new Promise((resolve) => {
    setTimeout(() => resolve(1), 1000);
}).finally(() => console.log(Date.now()));
```


可以看出，`Promise.prototype.finally` 的执行与 promise 实例的状态无关，不依赖于 promise 的执行后返回的结果值。其传入的参数是函数对象。


### 代码实现


实现思路：

- 考虑到 promise 的 resolver 可能是个异步函数，因此 finally 实现中，要通过调用实例上的 then 方法，添加 callback 逻辑
- 成功透传 value，失败透传 error

```typescript
Promise.prototype.finally = function (cb) {
    return this.then(
        (value) => Promise.resolve(cb()).then(() => value),
        (error) =>
            Promise.resolve(cb()).then(() => {
                throw error;
            })
    );
};
```


## 参考链接

- 文中的代码和用例均在：[github.com/dongyuanxin/diy-promise](https://github.com/dongyuanxin/diy-promise)
- [《ECMAScript 6 入门-Promise 对象》](http://es6.ruanyifeng.com/#docs/promise)
- [github.com/tc39/proposal-promise-allSettled](https://github.com/tc39/proposal-promise-allSettled)
- [github.com/matthew-andrews/Promise.prototype.finally](https://github.com/matthew-andrews/Promise.prototype.finally)

