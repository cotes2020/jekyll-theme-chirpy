---
title: "让我们再聊聊Promise的实现"
date: 2019-11-25
permalink: /2019-11-25-promise-a-plus/
categories: ["B源码精读", "Promise专题"]
---

## 摘要


关于 Promise 的实现网上已经有很多文章，最推荐的是这篇[《史上最易读懂的 Promise/A+ 完全实现》](https://zhuanlan.zhihu.com/p/21834559)。但其中`then`实现的部分代码中有些小问题。


在上述文章的基础上，本篇给出更全面的用例和代码，来尽可能阐述清楚一些看起来 "magic" 的做法。


所有代码（Promise 以及相关方法）的实现存放在 [github.com/dongyuanxin/diy-promise](https://github.com/dongyuanxin/diy-promise)。关于`.all()`，`.race()`等方法的实现，请见[《手写 Promise 的相关方法》](https://dongyuanxin.github.io/2019-11-23-promise-methods/)一文。


## 构造函数


### promise 实例的状态和值


先来看一段超级简单的代码：


```typescript
new Promise((resolve, reject) => {
  console.log("promise 1");
  resolve(1);
});
```


我们向 promise 构造函数中传入一个函数，这个函数的两个参数分别是：`resolve`和`reject`。显然，**这两个参数本身也是函数**。他们的作用就是**改变 promise 实例的状态**，resolve 是`pending -> resolved`，reject 是`pending -> rejected`。


除此之外，输出了`promise 1`，说明 promise 构造函数是立即执行的。


请看下面这段代码：


```typescript
new Promise((resolve, reject) => {
  reject(new Error("emypty"));
}).then(
  (value) => console.log("value is", value),
  (error) => console.log("error.message is", error.message)
);
// output: error.message is emypty
```


我们知道每个 promise 实例中，都有一个值（姑且称为 value）。当状态变为 resolved 时候，值就是传给 resolve()的参数；如果状态变为 rejected 时候，值就是传给 reject()的参数。也就是说，**实例的值在状态改变的时候才会有意义，并且不会再被改变**（因为状态只能变化一次）。


### 代码实现


下面，实现一个 promise 的构造函数。为了方便解释，关于 resolve 和 reject 的实现，放在下一节。


```typescript
// 3中状态
const STATUS_PENDING = Symbol("pending");
const STATUS_RESOLVED = Symbol("resolved");
const STATUS_REJECTED = Symbol("rejected");
function MyPromise(executor) {
  const that = this;
  that.data = null;
  that.status = STATUS_PENDING;
  // TODO: 下一节解释，promise实例的回调函数集
  that.onResolvedCallback = [];
  that.onRejectedCallback = [];
  function resolve(value) {
    // TODO: 下一节实现
  }
  function reject(error) {
    // TODO: 下一节实现
  }
  try {
    // 默认是交给参数传入的回调函数来执行状态的变更
    executor(resolve, reject);
  } catch (error) {
    // 如果回调函数的执行发生错误，抛出异常
    // promise会“主动”扭转状态
    reject(error);
  }
}
```


## resolve 和 reject 实现


resolve 和 reject 的实现相似，这里以 resolve 为例讲解。


上一节的例子中已经提到，resolve 的两个作用就是：

- 改变 promise 状态
- 更新 promise 的值

那么，根据逻辑，可以写出如下代码（当然，它是不完善的）：


```typescript
function resolve(value) {
  // 防止使用者先调用 reject，再调用resolve
  if (that.status !== STATUS_PENDING) {
    return;
  }
  that.data = value;
  that.status = STATUS_RESOLVED;
}
```


上一节的代码例子中，我们传给构造函数的参数，都不包含异步执行逻辑。如果包含异步执行，如下面的例子：


```text
new Promise((resolve) => {
  setTimeout(() => resolve(1), 1000);
}).then((value) => console.log("value is", value));

```


promise 实例的状态，在 1s 之后才会被扭转为 resolved。那么，此时传给`.then`的逻辑不应该执行，而是**保存下来，等到状态扭转后再执行**。因此，我们在 promise 的构造函数中用`onResolvedCallback`数组来保存 resolve 的回调函数集。


一个更完善的 resolve 实现：


```typescript
function resolve(value) {
  if (that.status !== STATUS_PENDING) {
    return;
  }
  that.data = value;
  that.status = STATUS_RESOLVED;
  for (let callback of that.onResolvedCallback) {
    // 这里that指的是指向promise实例的指针
    callback(that.data);
  }
}
```


同理，reject 的实现我们也可以写出来：


```typescript
function reject(error) {
  if (that.status !== STATUS_PENDING) {
    return;
  }
  that.data = error;
  that.status = STATUS_REJECTED;
  for (let callback of that.onRejectedCallback) {
    callback(that.data);
  }
}
```


## .then 的实现


根据[Promsie/A+规范](https://promisesaplus.com/)，`.then`方法返回一个新的 promise 实例。


根据 Promise 的 TS 类型提示，它接受两个函数参数：`onfulfilled` 和 `onrejected`。分别在 promise 实例的状态变为`resolved`和`rejected`时候调用。同时，**它们是可选参数**，意味着需要手动处理非函数类型的情况。


```typescript
MyPromise.prototype.then = function (onfulfilled, onrejected) {
  const that = this;
  onfulfilled =
    typeof onfulfilled === "function"
      ? onfulfilled
      : function (v) {
          return v; // 用于连续调用.then时，不传入onfulfilled情况下，借助“值穿透”来保证正常执行
        };
  onrejected =
    typeof onrejected === "function"
      ? onrejected
      : function (reason) {
          throw reason; // 如上同理，值穿透
        };
  if (that.status === STATUS_RESOLVED) {
    // TODO1
  }
  if (that.status === STATUS_REJECTED) {
    // TODO2
  }
  if (that.status === STATUS_PENDING) {
    // TODO3
  }
};
```


### TODO1：状态为 resolved 时


promise 状态变为 resolved，说明要执行`.then`的第一个 onfulfilled 函数参数，并且将 promise 的值作为参数传入 onfulfilled。


为了方便说明，`.then()`返回的 Promise 实例是 promise2，onfulfilled 函数返回值是 value。

1. 如果 value 不是 promise 类型，那么 promise2 的状态则更新为 resolved。
2. 如果 value 是 Promise 类型，那么调用 value 上的`then()`方法，来扭转状态，并且传入 promise2 构造函数的 resolve 和 reject，扭转 promise2

基于以上理解，可以写出 TODO1 的一版代码：


```typescript
const promise2 = new MyPromise((resolve, reject) => {
  try {
    const value = onfulfilled(that.data);
    if (value instanceof MyPromise) {
      value.then(resolve, reject);
    }
    resolve(value);
  } catch (error) {
    reject(error);
  }
});
return promise2;
```


请注意`value.then(resolve, reject);`这一句，由于没有`else`分之，会有人有疑问：这样可能会调用 2 次 promise2 的构造函数的 resolve？但其实这个担心是多余的，根据第一节封装的 resolve 逻辑，其内部会检查 promise 状态，不会多次重复。


这也是[《史上最易读懂的 Promise/A+ 完全实现》](https://zhuanlan.zhihu.com/p/21834559)文中的一个实现。**但是，它还是有问题**！这个问题比较难发现。


假设我们这样封装了，来看一下下面这段代码：


```typescript
new MyPromise((resolve, reject) => {
  resolve(1);
})
  .then((value) => {
    return new MyPromise((resolve) => {
      setTimeout(() => {
        resolve(value + 1);
      }, 1000);
    });
  })
  .then((value) => {
    console.log("final value is", value);
  });
```


理想情况下，它的输出应该是：


```text
final value is 2
```


但其实它的输出是：


```typescript
final value is MyPromise {
  data: null,
  status: Symbol(pending),
  onResolvedCallback: [ [Function] ],
  onRejectedCallback: [ [Function] ] }
```


问题就出现在前面没有`else`分支。虽然`value.then(resolve, reject)`和紧随其后`resolve(value);`语句，由于`function resolve()`有状态判断，不会调用两次。但如果在`.then()`返回的 promise2 实例的构造函数中，异步执行了状态扭转，那么，promise2 的状态就不是 resolved，而是 pending。此时，不会执行 TODO1 逻辑中的`value.then(resolve, reject)`，而是执行紧随其后`resolve(value);`语句。


因此，这就相当于直接将 value 传给了下一个`.then()`。所以，正确的 TODO1 的逻辑是：


```typescript
const promise2 = new MyPromise((resolve, reject) => {
  try {
    const value = onfulfilled(that.data);
    if (value instanceof MyPromise) {
      value.then(resolve, reject);
    } else {
      resolve(value);
    }
  } catch (error) {
    reject(error);
  }
});
return promise2;
```


### TODO2：状态为 rejected 时


在封装前，我们来看下真正的 promise 的行为：


```typescript
new Promise((resolve, reject) => {
  resolve(2 / abc);
})
  .then(
    (value) => {
      console.log("value is", value);
    },
    (error) => {
      console.log("error is", error.message);
      return 1;
    }
  )
  .then(
    (data) => {
      console.log("data is", data);
    },
    (error) => {
      console.log("error2 is", error);
    }
  );
// output:
// error is abc is not defined
// data is 1

```


可以看到，在第一个`then`中捕获了错误，并且进行处理。**此时调用并没有终止**，而是将返回值给了第二个`then`。


那如果在`then`中的错误处理函数，本身抛出了错误，会怎么样？


```typescript
new Promise((resolve, reject) => {
  resolve(2 / abc); // reject
})
  .then(null, (error) => {
    console.log("error is", 1 / abc); // throw error
    return 1;
  })
  .then(null, (error) => {
    console.log("error2 is", error);
  });
// output: error2 is abc is not defined

```


可以看到，第一个`then`错误处理函数抛出错误，传给了第二个`then`来处理。


以上的两个例子说明，TODO2 的实现和 TODO1 的类似，TODO2 逻辑的实现代码如下：


```typescript
return new MyPromise((resolve, reject) => {
  try {
    const value = onrejected(that.data);
    if (value instanceof MyPromise) {
      value.then(resolve, reject);
    } else {
      resolve(error);
    }
  } catch (error) {
    reject(error);
  }
});

```


### TODO3：状态为 pending 时


如果传给构造函数是个异步函数，那么实例的状态可能没有“来得及”变化，依然为`pending`。


还记得前面在编写构造函数的时候，留空的回调函数集吗？现在就派上用场了。这里将`onfulfilled`和`onrejected`这两种逻辑，分别放入`onResolvedCallback`和`onRejectedCallback`回调函数集中。


代码逻辑和 TODO1、TODO2 部分相似。


```typescript
return new MyPromise((resolve, reject) => {
  that.onResolvedCallback.push(function () {
    try {
      const value = onfulfilled(that.data);
      if (value instanceof MyPromise) {
        value.then(resolve, reject);
      } else {
        resolve(value);
      }
    } catch (error) {
      reject(error);
    }
  });
  that.onRejectedCallback.push(function () {
    try {
      if (value instanceof MyPromise) {
        value.then(resolve, reject);
      } else {
        resolve(value);
      }
    } catch (error) {
      reject(error);
    }
  });
});

```


## .catch 的实现


Promise 只有`.then()`方法。`.catch()`和`then()`相比，少了`onfulfilled`参数。


```typescript
MyPromise.prototype.catch = function (onrejected) {
  return this.then(null, onrejected);
};

```


## 参考

- [《史上最易读懂的 Promise/A+ 完全实现》](https://zhuanlan.zhihu.com/p/21834559)
- [Promsie/A+ Standard](https://promisesaplus.com/)

