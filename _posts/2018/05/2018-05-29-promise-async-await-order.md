---
title: "谈谈promise/async/await的执行顺序与V8引擎的BUG"
date: "2018-05-29"
permalink: /2018-05-29-promise-async-await-order/
categories: ["C工作实践分享"]
---

## 1. 题目和答案

> 故事还是要从下面这道面试题说起：请问下面这段代码的输出是什么？

```javascript
console.log("script start");

async function async1() {
  await async2();
  console.log("async1 end");
}

async function async2() {
  console.log("async2 end");
}
async1();

setTimeout(function() {
  console.log("setTimeout");
}, 0);

new Promise(resolve => {
  console.log("Promise");
  resolve();
})
  .then(function() {
    console.log("promise1");
  })
  .then(function() {
    console.log("promise2");
  });

console.log("script end");
```

上述，在`Chrome 66`和`node v10`中，正确输出是：

```bash
script start
async2 end
Promise
script end
promise1
promise2
async1 end
setTimeout
```

> **注意**：在新版本的浏览器中，`await`输出顺序被“提前”了，请看官耐心慢慢看。

## 2. 流程解释

边看输出结果，边做解释吧：

1. 正常输出`script start`
2. 执行`async1`函数，此函数中又调用了`async2`函数，输出`async2 end`。回到`async1`函数，**遇到了`await`，让出线程**。
3. 遇到`setTimeout`，扔到**下一轮宏任务队列**
4. 遇到`Promise`对象，立即执行其函数，输出`Promise`。其后的`resolve`，被扔到了微任务队列
5. 正常输出`script end`
6. 此时，此次`Event Loop`宏任务都执行完了。来看下第二步被扔进来的微任务，因为`async2`函数是`async`关键词修饰，因此，将`await async2`后的代码扔到微任务队列中
7. 执行第 4 步被扔到微任务队列的任务，输出`promise1`和`promise2`
8. 执行第 6 步被扔到微任务队列的任务，输出`async1 end`
9. 第一轮 EventLoop 完成，执行第二轮 EventLoop。执行`setTimeout`中的回调函数，输出`setTimeout`。

## 3. 再谈 async 和 await

细心的朋友肯定会发现前面第 6 步，如果`async2`函数是没有`async`关键词修饰的一个普通函数呢？

```javascript
// 新的async2函数
function async2() {
  console.log("async2 end");
}
```

输出结果如下所示：

```bash
script start
async2 end
Promise
script end
async1 end
promise1
promise2
setTimeout
```

不同的结果就出现在前面所说的第 6 步：如果 await 函数后面的函数是普通函数，那么其后的微任务就正常执行；否则，会将其再放入微任务队列。

## 4. 其实是 V8 引擎的 BUG

看到前面，正常人都会觉得真奇怪！（但是按照上面的诀窍倒也是可以理解）

然而 V8 团队确定了**这是个 bug**（很多强行解释要被打脸了），具体的 PR[请看这里](https://github.com/tc39/ecma262/pull/1250)。好在，这个问题已经在最新的 Chrome 浏览器中**被修复了**。

简单点说，前面两段不同代码的运行结果都是：

```bash
script start
async2 end
Promise
script end
async1 end
promise1
promise2
setTimeout
```

`await`就是让出线程，其后的代码放入微任务队列（不会再多一次放入的过程），就这么简单了。
