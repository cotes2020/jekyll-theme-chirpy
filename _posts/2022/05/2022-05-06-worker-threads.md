---
title: "NodeJS 多线程编程"
url: "2022-05-06-worker-threads"
date: 2022-05-06
---

> 开发环境：Node.JS v14.8.0


	整理自之前的《NodeJS多线程通信和共享内存 - worker_threads》&&《NodeJS多线程模块研究 - worker_threads》两篇文章。


## 快速开始


js 和 nodejs 一直都是单线程，直到官方推出了 worker_threads 模块，用来解决 CPU 密集型计算场景。


可以通过以下代码快速开启一个工作线程：


```javascript
if (isMainThread) {
    // 这会在工作线程实例中重新加载当前文件。
    new Worker(__filename);
    console.log("在主进程中");
    console.log("isMainThread is", isMainThread);
    // Worker可以启动指定文件，也可以配合eval参数，直接启动代码
    // 由于在子线程中没有process.env，所以这里可以配合workerData选项
    // 那么在子线程中，workerData会按照HTML结构化克隆算法
    // 将workerData克隆到 require('worker_thread')中
} else {
    console.log("在工作线程中");
    console.log("isMainThread is", isMainThread); // 打印 'false'。
}
```


上面代码的输出是：


```text
在主进程中
isMainThread is true
在工作线程中
isMainThread is false
```


## 多线程通信


### 父子线程通信-parentPort


master thread 中，`Worker` 返回的对象，代表着工作线程实例。


worker thread 中，`require('worker_threads').parentPort` 就是 master thread 的 MessagePort。


利用上面说到的两个对象，可以实现主线程和工作线程之间的“双工通信”。


```javascript
const { Worker, isMainThread, parentPort } = require("worker_threads");

// 使用parentPort和worker进行双工通信
if (isMainThread) {
    const worker = new Worker(__filename);
    // 1. 针对工作线程（worker）开启消息监听
    worker.on("message", (message) => console.log(message));
    // 2. 像工作线程（worker）发送消息："ping"
    worker.postMessage("ping");
} else {
    // 3. 针对主线程（parentPort）开启消息监听
    parentPort.on("message", (message) =>
        // 4. 接收到主线程的消息后，将其放入 pong 字段，并且发送给主线程（parentPort）
        parentPort.postMessage({ pong: message })
    );
}

```


上面输出：


```text
{
    pong: "ping";
}
// handling
```


> node 的设计参考了浏览器的 Web Worker


### 任意线程通信-MessageChannel


通信管道（MessageChannel）可以跨任何线程（比如多个工作线程之间）进行数据通信。对比上面通过`require('worker_threads').parentPort` 的方式，缺点是只能在主线程和工作线程之间进行通信。


管道通信的思路：

- 通过 `MessageChannel` 创建通信管道
- 通过 `postMessage` 传递数据
	- 第一个参数是传递的数据，包括通信管道的 `Messageport`
	- 第二个参数是 `transformList`，放入其中的对象，将在管道发送端中无法使用

来看一段利用通信管道，兄弟线程通信的实现：


```javascript
const {
    isMainThread, parentPort, threadId, MessageChannel, Worker
} = require('worker_threads');

if (isMainThread) {
    const worker1 = new Worker(__filename);
    const worker2 = new Worker(__filename);
    const subChannel = new MessageChannel();
    worker1.postMessage({ yourPort: subChannel.port1 }, [subChannel.port1]);
    worker2.postMessage({ yourPort: subChannel.port2 }, [subChannel.port2]);
} else {
    parentPort.once('message', (value) => {
        value.yourPort.postMessage('hello');
        value.yourPort.on('message', msg => {
            console.log(`thread ${threadId}: receive ${msg}`);
        });
    });
}
```


控制台输出：


```text
thread 1: receive hello
thread 2: receive hell
```


### 共享内存通信-SharedArrayBuffer


nodejs 多线程也可以通过“共享内存”进行通信。master thread 和 worker threads 都操作同一段物理存储上的数据，实现数据共享，并且避免了拷贝带来的开销。


共享内存通信的思路：

- 通过 `SharedArrayBuffer` 创建二进制数据
- `SharedArrayBuffer` 数据不能放在 `postMessage` 的第二个参数中

来看一段通过共享内存进行通信的代码：


```javascript
const assert = require("assert");
const {
    Worker,
    MessageChannel,
    MessagePort,
    isMainThread,
    parentPort,
} = require("worker_threads");
if (isMainThread) {
    const worker = new Worker(__filename);
    const subChannel = new MessageChannel();
    // uint8Arr 是放在共享内存上的
    const sharedArray = new SharedArrayBuffer(4);
    const uint8Arr = new Uint8Array(sharedArray);
    console.log("[master] 原来的uint8Arr", uint8Arr);

    worker.postMessage({ hereIsYourPort: subChannel.port1, uint8Arr }, [
        subChannel.port1,
    ]);

    subChannel.port2.on("message", (msg) => {
        console.log("[master] 经过工作线程修改的uint8Arr", uint8Arr);
    });
} else {
    parentPort.on("message", (value) => {
        assert(value.hereIsYourPort instanceof MessagePort);
        value.uint8Arr[1] = 10;
        console.log("[worker] 修改出来的uint8Arr", value.uint8Arr);
        value.hereIsYourPort.postMessage("");
        value.hereIsYourPort.close();
    });
}

```


上面代码的输出是：


```text
[master] 原来的uint8Arr Uint8Array(4) [ 0, 0, 0, 0 ]
[worker] 修改出来的uint8Arr Uint8Array(4) [ 0, 10, 0, 0 ]
[master] 经过工作线程修改的uint8Arr Uint8Array(4) [ 0, 10, 0, 0 ]

```


可以看到，worker thread 修改了 uint8Arr，由于采用共享内存，因此在 master thread 中，从自己的上下文中，也读取到修改后的 uint8Arr。


## worker_threads 效果评测


### 单线程计算


以一个生成素数的计算为例，直接跑2-1e7范围内的素数：


```javascript
const min = 2;
const max = 1e7;
const primes = [];
function generatePrimes(start, range) {
  let isPrime = true;
  let end = start + range;
  for (let i = start; i < end; i++) {
    for (let j = min; j < Math.sqrt(end); j++) {
      if (i !== j && i%j === 0) {
        isPrime = false;
        break;
      }
    }
    if (isPrime) {
      primes.push(i);
    }
    isPrime = true;
  }
}
generatePrimes(min, max);

```


通过 `time` 命令能看到，这里一共耗时10.36s，单核cpu占用率达到了99%


![e6c9d24egy1h1zys3l4rtj20ny01uq38.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-05-06-worker-threads/e6c9d24egy1h1zys3l4rtj20ny01uq38.jpg)


### 工作线程计算


主线程负责切割数据，将数据等分后，下发给工作线程。


工作线程从 workerData 上，读取计算范围，同时计算范围内的素数。


代码如下：


```javascript
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const min = 2;
let primes = [];
function generatePrimes(start, range) {
  let isPrime = true;
  let end = start + range;
  for (let i = start; i < end; i++) {
    for (let j = min; j < Math.sqrt(end); j++) {
      if (i !== j && i%j === 0) {
        isPrime = false;
        break;
      }
    }
    if (isPrime) {
      primes.push(i);
    }
    isPrime = true;
  }
}
if (isMainThread) {
  const max = 1e7;
  const threadCount = +process.argv[2] || 4;
  const threads = new Set();;
  console.log(`Running with ${threadCount} threads...`);
  const range = Math.ceil((max - min) / threadCount);
  let start = min;
  for (let i = 0; i < threadCount - 1; i++) {
    const myStart = start;
    threads.add(new Worker(__filename, { workerData: { start: myStart, range }}));
    start += range;
  }
  threads.add(new Worker(__filename, { workerData: { start, range: range + ((max - min + 1) % threadCount)}}));
  for (let worker of threads) {
    worker.on('error', (err) => { throw err; });
    worker.on('exit', () => {
      threads.delete(worker);
      console.log(`Thread exiting, ${threads.size} running...`);
      if (threads.size === 0) {
        // console.log(primes.join('\\n'));
      }
    })
    worker.on('message', (msg) => {
      primes = primes.concat(msg);
    });
  }
} else {
  generatePrimes(workerData.start, workerData.range);
  parentPort.postMessage(primes);
}

```


在本地开启4个工作线程，并且本地6核CPU不处于忙碌的状态，最终耗时是2.658s，缩短了4倍。CPU利用率是 314%，，调用了3+CPU资源来支持本次计算：


![e6c9d24egy1h1zyw3icekj20om05gwfa.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-05-06-worker-threads/e6c9d24egy1h1zyw3icekj20om05gwfa.jpg)


**可以看到，优化效果非常明显，能充分利用多核CPU的优势**。


## worker_threads 的底层模型


对于单线程，Nodejs 是由以下部分组成：

- 一个进程
- 一个线程
- 一个事件循环
- 一个 JS 引擎实例
- 一个 Node.js 实例

对于工作线程，Nodejs 组成变成了：

- 一个进程
- 多个线程
- 每个线程独立的事件循环
- 每个线程独立的 JS 引擎实例
- 每个线程独立的 Node.js 实例

如下图所示：


![0081Kckwgy1gkl73ybha8j31k90u0ada.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-05-06-worker-threads/0081Kckwgy1gkl73ybha8j31k90u0ada.jpg)


## Q & A


### 多线程是否可以并行？workder_threads 能否提高CPU利用率？


并发是快速切换任务，本质还是串行执行；并行是同时执行任务。


而多线程可以并发，也可以并行，这取决于是否能抢占到 CPU 资源。这个过程是操作系统来调度，不可人为控制。


但总的来说，多线程和多进程类似，都可以 CPU 的利用率。


### workder_threads和cluster/child_process的区别


child_process是nodejs的多进程模块。cluster是在child_process上封装的集群模块，提供了多进程并行下的各种API。**他们都是基于进程模型**。


worker_threads 基于线程模型，开销小，更加轻量，并且在同一个进程下，还可以通过共享内存来通信。**适合解决CPU密集问题**。


## 参考文章

- [Understanding Worker Threads in Node.js](https://nodesource.com/blog/worker-threads-nodejs/)
- [多线程 ---并发与并行概念总结](https://blog.csdn.net/qq_33290787/article/details/51790605)
- [Using worker_threads in Node.js](https://medium.com/@Trott/using-worker-threads-in-node-js-80494136dbb6)
- [Using worker_threads in Node.js Part 2](https://medium.com/@Trott/using-worker-threads-in-node-js-part-2-a9405c72a6f0)

