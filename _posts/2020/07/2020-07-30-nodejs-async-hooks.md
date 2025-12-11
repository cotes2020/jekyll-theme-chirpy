---
title: "Node.js async_hooks模块：实现异步上下文"
date: 2020-07-30
permalink: /2020-07-30-nodejs-async-hooks/
categories: ["C工作实践分享"]
tags: [异步编程, AsyncContext]
---
## 为什么需要 async_hooks？


node 是异步编程模型，存在一些痛点问题：

- 异步调用链很难追踪
- 缺少异步上下文的概念
- 缺少异步上下文的存储 API

而 async_hooks 就是为了解决这些问题而生。


## Async Scope


函数有上下文，异步调用也有上下文。


对于一段异步代码，在不做特殊处理情况下，无法得知是哪个函数调用了此段代码：


```typescript
function readFile(filePath) {
  fs.read(filePath, (err, data) => {
    console.log('data is', data)
  })
}
```


readFile 可能被 Function A、Async Function A 调用，也可能在 setTimeout 等回调函数中调用。


而每个 async scope 都有一个上下文，`executionAsyncId()`返回当前 async scope 的 id，`triggerAsyncId()`返回调用者的 async scope 的 id。


## Async Hooks


每次 async scope 生成或者销毁时，都会触发 async hook，可以通过`creatHook`创建相关 hook，并且启用。


```typescript
const fs = require('fs')
const async_hooks = require('async_hooks')
async_hooks.createHook({
  init (asyncId, type, triggerAsyncId, resource) {
    fs.writeSync(1, `${type}(${asyncId}): trigger: ${triggerAsyncId}\n`)
  },
  destroy (asyncId) {
    fs.writeSync(1, `destroy: ${asyncId}\n`);
  }
}).enable()
async function A () {
  fs.writeSync(1, `A -> ${async_hooks.executionAsyncId()}\n`)
  setTimeout(() => {
    fs.writeSync(1, `A in setTimeout -> ${async_hooks.executionAsyncId()}\n`)
    B()
  })
}
async function B () {
  fs.writeSync(1, `B -> ${async_hooks.executionAsyncId()}\n`)
  process.nextTick(() => {
    fs.writeSync(1, `B in process.nextTick -> ${async_hooks.executionAsyncId()}\n`)
    C()
    C()
  })
}
function C () {
  fs.writeSync(1, `C -> ${async_hooks.executionAsyncId()}\n`)
  Promise.resolve().then(() => {
    fs.writeSync(1, `C in promise.then -> ${async_hooks.executionAsyncId()}\n`)
  })
}
fs.writeSync(1, `top level -> ${async_hooks.executionAsyncId()}\n`)
A()
```


## CLS：Connection Local Storage


对于多线程的语言，例如 Java，有 TLS（Thread Local Storage）。它提供线程级存储，只能在相同线程内访问到。


类似地，对于异步模型的 Nodejs，CLS 提供 Async Scope 级的存储。它只能在异步上下文中被访问。


```typescript
const http = require('http');
const fs = require('fs')
const { AsyncLocalStorage } = require('async_hooks');

const asyncLocalStorage = new AsyncLocalStorage();

function logWithId(msg) {
  console.log('logWithId: ', asyncLocalStorage.getStore())
}

function readFile() {
    fs.readFile('./index.js', (err, data) => {
        console.log('readFile: ', asyncLocalStorage.getStore())
    })
}

let idSeq = 0;
http.createServer((req, res) => {
  asyncLocalStorage.run(idSeq++, () => {
    logWithId(); // 正常打印
    readFile(); // 正常打印
    setImmediate(() => {
      logWithId(); // 正常打印
      res.end();
    });
  });

  logWithId(); // 打印：undefined
  readFile(); // 打印：undefined
}).listen(8080);

http.get('http://localhost:8080');
```


## 应用场景


应用 1：利用 Hook 接口，追踪异步调用链路


应用 2：利用 CLS，埋点请求链路信息


对于应用 2，类似于 Nestjs 提供的 Scope.REQUEST 概念：对每次请求，生成实例。


对于 NestJS/Koa/Express 等 Node 应用来说，都有`next`回调函数的概念，由开发者控制。


**因此，可以在请求进入的最前面的中间件中，创建整个链路都会使用的信息（例如 RequestID），并且构造 CLS**。


```typescript
let alsId = 0;

async function firstMiddleware(ctx, next) {
  asyncLocalStorage.run(alsId++, () => next());
}
```


这是一个简单的 demo，之后的链路中，都可以通过 asyncLocalStorage.getStore() 来拿到请求 ID。


在真实场景中，store 可以是一个复杂对象。


## 参考链接

- 学习使用 Node.js 中的 async-hooks 模块：[https://zhuanlan.zhihu.com/p/53036228](https://zhuanlan.zhihu.com/p/53036228)
- Node v14：[https://nodejs.org/dist/latest-v14.x/docs/api/async_hooks.htm](https://nodejs.org/dist/latest-v14.x/docs/api/async_hooks.htm)
- RequestId Tracing in Node.js Applications：[https://itnext.io/request-id-tracing-in-node-js-applications-c517c7dab62d](https://itnext.io/request-id-tracing-in-node-js-applications-c517c7dab62d)
- One Node.JS CLS API to rule them all：[https://itnext.io/one-node-js-cls-api-to-rule-them-all-1670ac66a9e8](https://itnext.io/one-node-js-cls-api-to-rule-them-all-1670ac66a9e8)

