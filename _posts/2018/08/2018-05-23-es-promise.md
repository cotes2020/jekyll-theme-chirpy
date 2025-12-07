---
title: "Promise 概述"
date: 2018-05-23
permalink: /2018-05-23-es-promise/
categories: ["源码精读", "Promise专题"]
---

## 关于`Promise`

- `Promise`实例一旦被创建就会被执行
- `Promise`过程分为两个分支：`pending=>resolved`和`pending=>rejected`
- `Promise`状态改变后，依然会执行之后的代码：

```javascript
const warnDemo = ctx => {
  const promise = new Promise(resolve => {
    resolve(ctx);
    console.log("After resolved, but Run"); // 依然会执行这个语句
  });
  return promise;
};

warnDemo("ctx").then(ctx => console.log(`This is ${ctx}`));
```

## `then`方法

在`Console`键入以下内容：

```javascript
let t = new Promise(() => {});
t.__proto__;
```

可以看到，`then`方法是定义在原型对象`Promise.prototype`上的。

`then`方法的第一个参数是`resolved`状态的回调函数，第二个参数（可选）是`rejected`状态的回调函数。

#### 写法

```javascript
function func(args) {
    // 必须返回一个Promise实例
    return new Promise((resolve,reject)=>{
        if(...){
            resolve(...) // 传入resolve函数的参数
        } else {
            let err = new Error(...)
            reject(err) // reject参数必须是Error对象
        }
    })
}

func(ARGS).then(()=>{
    // resolve 函数
},()=>{
    // reject 函数
})
```

#### 连续调用`then`

因为`then`方法返回另一个`Promise`对象。当这个对象状态发生改变，就会分别调用`resolve`和`reject`

写法如下：

```javascript
func(ARGS).then(()=>{
    ...
}).then(
    ()=>{ ... },
    () => { ... }
)
```

#### 实例

```javascript
function helloWorld(ready) {
  return new Promise((resolve, reject) => {
    if (ready) {
      resolve("Right");
    } else {
      let error = new Error("arg is false");
      reject(error); // 传入Error对象
    }
  });
}

helloWorld(false).then(
  msg => {
    // true：helloWorld的参数
    // 参数msg：在上面的Promise对象中传入了
    console.log(msg);
  },
  error => {
    console.log(error.message);
  }
);
```

## `catch`方法

等同于 `.then(null, rejection)`。另外，`then`方法指定的回调函数运行中的错误，也会被`catch`捕获。

所以，之前的写法可以改为：

```javascript
function func(args) {
    // 必须返回一个Promise实例
    const promise =  new Promise((resolve,reject)=>{
        if(...){
            resolve(...)
        } else {
            let err = new Error(...)
            reject(err)
        }
    })
    return promise
}

func(ARGS).then(()=>{
    // resolve 函数
}).catch(()=>{
    // reject 函数
}).then(()=>{
    // 没有错误就会跳过上面的catch
})...
```

## `finally`方法

> 指定不管 `Promise` 对象最后状态如何，都会执行的操作。可以理解为`then`方法的实例，即在`resolve`和`reject`里面的公共操作函数

## `all`方法

> 用于将多个 `Promise` 实例，包装成一个新的 `Promise` 实例。它接收一个具有`Iterator`接口的参数。其中，`item`如果不是`Promise`对象，会自动调用`Promise.resolve`方法

以下代码：

```javascript
const p = Promise.all([p1, p2, p3]); // p是新包装好的一个Promise对象
```

对于`Promise.all()`包装的`Promise`对象，只有实例的状态都变成`fulfilled`。

可以用来操作数据库：

```javascript
const databasePromise = connectDatabase();

const booksPromise = databasePromise.then(findAllBooks);

const userPromise = databasePromise.then(getCurrentUser);

Promise.all([booksPromise, userPromise]).then(([books, user]) =>
  pickTopRecommentations(books, user)
);
```

或者其中有一个变为`rejected`，才会调用`Promise.all`方法后面的回调函数。而对于每个`promise`对象，一旦它被自己定义`catch`方法捕获异常，那么状态就会更新为`resolved`而不是`rejected`。

```javascript
"use strict";
const p1 = new Promise((resolve, reject) => {
  resolve("hello");
})
  .then(result => result)
  .catch(e => e);

const p2 = new Promise((resolve, reject) => {
  throw new Error("p2 error");
})
  .then(result => result)
  .catch(
    // 如果注释掉 catch，进入情况2
    // 否则，情况1
    e => e.message
  );

Promise.all([p1, p2])
  .then(
    result => console.log(result) // 情况1
  )
  .catch(
    e => console.log("error in all") // 情况2
  );
```

## `race方法`

> 和`all`方法类似，`Promise.race`方法同样是将多个 `Promise` 实例，包装成一个新的 `Promise` 实例。而且只要有一个状态被改变，那么新的`Promise`状态会立即改变

也是来自阮一峰大大的例子，如果 5 秒内无法`fetech`，那么`p`状态就会变为`rejected`。

```javascript
const p = Promise.race([
  fetch("/resource-that-may-take-a-while"),
  new Promise(function(resolve, reject) {
    setTimeout(() => reject(new Error("request timeout")), 5000);
  })
]);
p.then(response => console.log(response));
p.catch(error => console.log(error));
```

## 重要性质

### 状态只改变一次

> `Promise` 的状态一旦改变，就永久保持该状态，不会再变了。

下面代码中，`Promise`对象`resolved`后，状态就无法再变成`rejected`了。

```javascript
"use strict";

const promise = new Promise((resolve, reject) => {
  resolve("ok"); // 状态变成 resolved
  throw new Error("test"); // Promise 的状态一旦改变，就永久保持该状态
});
promise
  .then(val => {
    console.log(val);
  })
  .catch(error => {
    console.log(error.message); // 所以，无法捕获错误
  });
```

### 错误冒泡

> `Promise` 对象的错误具有“冒泡”性质，**会一直向后传递，直到被捕获为止**。也就是说，错误总是会被下一个`catch`语句捕获

### "吃掉错误"机制

> `Promise`会吃掉内部的错误，并不影响外部代码的运行。所以需要`catch`，以防丢掉错误信息。

阮一峰大大给出的 demo：

```javascript
"use strict";

const someAsyncThing = function() {
  return new Promise(function(resolve, reject) {
    // 下面一行会报错，因为x没有声明
    resolve(x + 2);
  });
};

someAsyncThing().then(function() {
  console.log("everything is great");
});

setTimeout(() => {
  console.log(123);
}, 2000);
```

还有如下 demo

```javascript
someAsyncThing()
  .then(function() {
    return someOtherAsyncThing();
  })
  .catch(function(error) {
    console.log("oh no", error);
    // 下面一行会报错，因为y没有声明
    y + 2;
  })
  .catch(function(error) {
    console.log("carry on", error);
  });
// oh no [ReferenceError: x is not defined]
// carry on [ReferenceError: y is not defined]
```

## 参考

- demo 基本可以在[阮一峰的 Es6 讲解](http://es6.ruanyifeng.com/#docs/promise)中找到，只是为了理解做了一些修改。
- 还有网上的一些博客，这里就不一一说明了
