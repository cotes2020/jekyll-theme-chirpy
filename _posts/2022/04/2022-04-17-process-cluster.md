---
title: "NodeJS 多进程编程"
url: "2022-04-17-process-cluster"
date: 2022-04-17
---

## nodejs中的进程对象-process


### 常见属性和方法


在nodejs中，进程就是process对象。在这个对象上有一些常用的属性和方法：

- env: 环境变量
- pid/ppid: 进程id/父进程id
- `cwd()`/`chdir(directory)`: 当前执行路径/修改执行路径
- argv/execArgv: 给JS脚本的参数/NodeJS的参数
- stdin/stdout/stderr: 标准输入/标准输出/标准错误

### 如何处理为捕获的异常？


Nodejs 可以通过 try-catch 来捕获异常。如果异常未捕获，则会一直从底向事件循环冒泡。如是冒泡到事件循环的异常没被处理，那么就会导致当前进程异常退出。


根据文档，可以通过监听 process 的 uncaughtException 事件，来处理未捕获的异常：


```javascript
process.on("uncaughtException", (err, origin) => {
    console.log(err.message);
});
const a = 1 / b;
console.log("abc"); // 不会执行

```


上面的代码，控制台的输出是：b is not defined。捕获了错误信息，并且进程以0退出。开发者可以在 uncaughtException 事件中，清除一些已经分配的资源（文件描述符、句柄等），不推荐在其中重启进程。


可以通过监听 unhandledRejection 事件，来处理未被捕获的Promise错误：


```javascript
process.on("unhandledRejection", (err, promise) => {
    console.log(err.message);
});
Promise.reject(new Error("错误信息")); // 未被catch捕获的异常，交由unhandledRejection事件处理

```


可以通过监听 warning 事件，来处理告警。告警不是 Node.js 和 Javascript 错误处理流程的正式组成部分。 一旦探测到可能导致应用性能问题，缺陷或安全隐患相关的代码实践，Node.js 就可发出告警。


### 如何处理进程退出？


### `exit()` VS `exitCode`


一个 nodejs 进程，可以通过 process.exit() 来指定退出代码，直接退出。**不推荐直接使用 process.exit()**，这会导致事件循环中的任务直接不被处理，以及可能导致数据的截断和丢失（例如 stdout 的写入）。


```javascript
setTimeout(() => {
    console.log("我不会执行");
});
process.exit(0);

```


**正确安全的处理是**，设置 process.exitCode，并允许进程自然退出。


```javascript
setTimeout(() => {
    console.log("我不会执行");
});
process.exitCode = 1;
```


### beforeExit 和 exit 事件


用于处理进程退出的事件有：beforeExit 事件 和 exit 事件。


当 Node.js 清空其事件循环并且没有其他工作要安排时，会触发 beforeExit 事件。例如在退出前需要一些异步操作，那么可以写在 beforeExit 事件中：


```javascript
let hasSend = false;
process.on("beforeExit", () => {
    if (hasSend) return; // 避免死循环
    setTimeout(() => {
        console.log("mock send data to serve");
        hasSend = true;
    }, 500);
});
console.log(".......");
// 输出：
// .......
// mock send data to serve

```


> 注意：在 beforeExit 事件中如果是异步任务，那么又会被添加到任务队列。此时，任务队列完成所有任务后，又会触发 beforeExit 事件。因此，不处理的话，可能出现死循环的情况。如果是显式调用 exit()，那么不会触发此事件。


在 exit 事件中，只能执行同步操作。在调用 'exit' 事件监听器之后，Node.js 进程将立即退出，从而导致在事件循环中仍排队的任何其他工作被放弃。


### 如何理解 process.nextTick？


我第一次看到 process.nextTick 的时候是比较懵的，看文档可以知道，它的用途是：把回调函数作为微任务，放入事件循环的任务队列中。但这么做的意义是什么呢？


因为 nodejs 并不适合计算密集型的应用，一个进程就一个线程，在当下时间点上，就一个事件在执行。那么，如果我们的事件占用了很多 cpu 时间，那么之后的事件就要等待非常久。所以，**nodejs 的一个编程原则是尽量缩短每一个事件的执行事件**。process.nextTick 的作用就在这，**将一个大的任务分解成多个小的任务**。示例代码如下：


```text
// 被拆分成2个函数执行
function BigThing() {
    doPartThing();
    process.nextTick(() => finishThing());
}
```


在事件循环中，何时执行 nextTick 注册的任务呢？请看下面的代码：


```javascript
setTimeout(function() {
    console.log("第一个1秒");
    process.nextTick(function() {
        console.log("第一个1秒：nextTick");
    });
}, 1000);
setTimeout(function() {
    console.log("第2个1秒");
}, 1000);
console.log("我要输出1");
process.nextTick(function() {
    console.log("nextTick");
});
console.log("我要输出2");
```


输出的结果如下，nextTick 是早于 setTimeout：


```text
我要输出1
我要输出2
nextTick
第一个1秒
第一个1秒：nextTick
第2个1秒
```


在浏览器端，nextTick 会退化成 `setTimeout(callback, 0)`。但在 nodejs 中请使用 nextTick 而不是 setTimeout，前者效率更高，并且严格来说，两者创建的事件在任务队列中顺序并不一样（请看前面的代码）。


### 如何处理信号量（signal）？


大多数操作系统通过信号量将消息发送给一个程序。


在nodejs中，对于 `process.kill(pid, [signal])` 函数，如果传入signal，那么不是杀死进程，而是向进程传递信号量。


在nodejs中，可以通过`process.on()`监听程序的信号量，并且做出响应：


```javascript
process.stdin.resume(); // 必须加，要不然程序会退出，因为没监听data时间，所以是可读流的暂停模式

process.on('SIGINT', function () {
  // Ctrl+C 强行终止命令，会发送 SIGINT 信号量
  console.log('Received SIGINT. Press Control-D to exit.');
});

console.log(`本进程的id是: ${process.pid}`); // 进程id，方便外界 kill

```


> 可以前往NodeJS Doc了解更多信号量


## nodejs中的子进程-child_process


在nodejs中，借助子进程模块，可以创建多进程。


### 如何创建多进程？


通过以下4个方法可以创建，并且均是异步的，而且返回一个 `ChildProcess` 实例。

- execFile
- spawn
- exec
- fork

> 在什么场景下使用他们？

- execFile：**当只需要执行应用程序并且获取输出的时候**。比如执行图像处理类的脚本，只关注是否成功，不用拿回大量的二进制数据。
- spawn：它的返回是一个基于Stream的对象。**适合处理产生/输入大量数据的应用程序**。同时，使用Stream，也有各种好处。
- exec：
	- 和execFile、spawn相比，没有`args`参数。
	- **可以一次性拼接多个命令以及它们的参数**。就像在shell中一样。
	- 在使用到管道、重定向、file glob的时候，默认会创建shell，效率更高。
- fork：
	- 底层是spawn实现，相对使用更便捷。
	- *在需要利用IPC通信的时候。**fork会打开一个IPC通道，可以在主子进程间传递消息
	- **需要快速执行一个计算进程，不想阻塞主进程**

代码示例见：[Understanding execFile, spawn, exec, and fork in Node.js](https://dzone.com/articles/understanding-execfile-spawn-exec-and-fork-in-node)


### 如何进行进程间通信？


在nodejs中，进程间通信主要有以下方式：


### stdio/stdout 管道


父进程实现：


```javascript
// parent.js
const { spawn } = require('child_process');

main();

function main() {
  const child = spawn('node', ['./child.js']);

  // 先处理错误输出
  child.stderr.pipe(process.stderr);

  // 监听子进程的输出，从而接收子进程消息
  child.stdout.on('data', function (chunk) {
    const str = chunk.toString('utf-8');
    try {
      const { payload, type } = JSON.parse(str);
      if (type === 'msg') {
        console.log(`(收到子进程消息)${payload}`);
      }
    } catch (err) {
      console.log('子进程普通输出:\\n' + str);
    }
  });

  sendMsgToChild(child, '你好，我是父进程');
}

// 向子进程传递消息
function sendMsgToChild(child, payload) {
  const str = JSON.stringify({
    type: 'msg',
    payload,
  });
  // 向子进程传递消息
  child.stdin.write(str);
}

```


子进程实现：


```javascript
process.stdin.on('data', (chunk) => {
  const str = chunk.toString('utf-8');
  try {
    const { payload, type } = JSON.parse(str);
    if (type === 'msg') {
      console.log(`(收到父进程消息)${payload}`);
      sendMsgToParent('你好，我是子进程');
    }
  } catch (err) {
    console.log(`(收到父进程输入)${str}`);
  }
});

function sendMsgToParent(payload) {
  const str = JSON.stringify({
    type: 'msg',
    payload,
  });
  console.log(str);
}

```


输出：


```text
子进程普通输出:
(收到父进程消息)你好，我是父进程

(收到子进程消息)你好，我是子进程
```


### NodeJS 内置 IPC


这个是NodeJS原生支持的IPC机制。通过`fork()`方式创建的子进程，可以使用。


父进程：


```javascript
const { fork } = require('child_process');
const child = fork('./child.js');

child.send('我是父进程');
child.on('message', (message) => {
  console.log('(来自子进程消息)' + message);
});

```


子进程：


```javascript
process.on('message', (msg) => {
  console.log('(来自父进程消息)' + msg);
  process.send('我是子进程');
});

```


### Socket 通信


在本机中架设一个TCP/UDP服务器，来作为本地的进程消息中转站。在 node-ipc 库中，实现了这一套机制，可以直接使用。


在当前进程中，创建一个socket服务：


```typescript
import ipc from 'node-ipc';

ipc.config.id = 'world';
ipc.config.retry = 1500;
ipc.config.maxConnections = 1;

ipc.serveNet(function () {
  ipc.server.on('message', function (data, socket) {
    ipc.log('>>> message : ', data);
    ipc.server.emit(socket, 'message', data + ' world!');
  });

  ipc.server.on('socket.disconnected', function (data, socket) {
    console.log('>>> socket.disconnected\\n\\n', 'arguments');
  });
});
ipc.server.on('error', function (err) {
  ipc.log('>>> error', err);
});
ipc.server.start();

```


在其它进程中，和已创建的socket服务进行通信：


```typescript
import ipc from 'node-ipc';

// ipc.config.id = 'hello';
ipc.config.retry = 1500;

ipc.connectToNet('world', function () {
  ipc.of.world.on('connect', function () {
    ipc.log('<<< connected to world', ipc.config.delay);
    ipc.of.world.emit('message', 'hello');
  });
  ipc.of.world.on('message', function (data) {
    ipc.log('<<< got a message from world : ', data);
  });

  ipc.of.world.on('error', (err) => {
    console.log('<<< err is', err);
  });
  ipc.of.world.on('disconnect', function () {
    ipc.log('<<< disconnected from world');
  });
});

```


### 中间件通信


通过redis/MQ等第三方中间件来进行进程间消息传递。不常用。


> NodeJS的进程通信是基于操作系统实现的。站在操作系统来说，常用的IPC有：无名管道、FIFO、（内存）消息队列、信号量以及共享内存。  
> 有空写篇文章，在NodeJS开发中，基本不需要关心操作系统的IPC，使用NodeJS包装好的IPC方法即可。


### 如何调度多进程？


按照《深入浅出 nodejs》，在处理 cpu 密集型问题的时候，应该使用 master/worker 编程模型，以充分利用现代计算机的多核优势。


但对于 nodejs 来说，每次进行计算都启动一个实例是非常浪费时间的（v8、加载库、开辟进程空间等等）。所以可以准备一个进程池，池中实例可以重复利用，并且支持排队操作。


这里需要手动实现一个多进程的池子，减少重复创建Worker带来的损耗。


本身采用的是 Master-Worker 架构：

- Master：负责调度Worker进程，收发消息
- Worker：负责执行具体工作逻辑

整体流程：

- Master 创建 Pool
- 使用者通过 Master 下发任务
- Master 内部对 Pool 中的 Worker 进行调度
- Worker 接收任务，并且执行。

Master实现：


```javascript
// <https://github.com/dongyuanxin/ciy/blob/master/nodejs/process/pool.js>
const cp = require('child_process');
const cpuNum = require('os').cpus().length; // 用CPU的核数作为Pool的最大容量

/**
 * 声明一个针对指定worker的进程池
 * @param {string} workModule
 * @return {function}
 */
function creatProcessPool(workModule) {
  const waitingQueue = []; // 任务等待队列
  const readyPool = []; // 可用的worker存放的池子
  let poolSize = 0; // 池的大小 = 可用的worker + 正在使用中worker

  /**
   * 将信号发送给池中可用的worker
   * @param {string} job 任务信号
   * @param {function} callback
   */
  return function doWork(job, callback) {
    callback = callback || (() => {});

    // 如果池中没有可用worker，且池的大小已经到上限
    if (!readyPool.length && poolSize > cpuNum) {
      waitingQueue.push([job, callback]);
      return;
    }

    let child = null;
    if (readyPool.length) {
      // 池中有可用worker
      child = readyPool.shift();
    } else {
      // 池中没有可用worker，并且当前worker还可以申请
      child = cp.fork(workModule);
      ++poolSize;
    }

    let cbTriggered = false; // 防止回调函数重复调用

    child
      .once('error', (err) => {
        if (!cbTriggered) {
          callback(err);
          cbTriggered = true;
        }
        child.kill();
      })
      .once('exit', (code) => {
        if (!cbTriggered) {
          callback(new Error('Worker exited with code:' + code));
        }
        --poolSize;
        const childIdx = readyPool.indexOf(child);
        readyPool.splice(childIdx);
      })
      .once('message', (msg) => {
        // 当worker完成cpu计算后
        // 发送消息给master，重新回收worker
        callback(null, msg);
        cbTriggered = true;
        readyPool.push(child);

        // 如果等待队列中还有未完成任务，则执行
        if (waitingQueue.length) {
          // 防止阻塞主线程
          setImmediate(() => {
            doWork(...waitingQueue.shift());
          });
        }
      })
      .send(job); // 向worker发送指令
  };
}

module.exports.creatProcessPool = creatProcessPool;

```


一个简单的 Worker 的demo：


```javascript
// <https://github.com/dongyuanxin/ciy/blob/master/nodejs/process/pool.worker.js>
const map = {
    'A': handleJobA,
    'B': handleJobB
}

process.on('message', (msg) => {
    if (map[msg]) {
        const result = map[msg]()
        process.send(result)
    } else {
        process.send('Job not exist')
    }
})

function handleJobA() {
    for (let i = 0; i < 1e10; i++){}
    for (let i = 0; i < 1e10; i++){}
    return 'handle job A'
}

function handleJobB() {
    for (let i = 0; i < 1e10; i++){}
    for (let i = 0; i < 1e10; i++){}
    return 'handle job B'
}

```


使用效果：


```javascript
// <https://github.com/dongyuanxin/ciy/blob/master/nodejs/process/pool.spec.js>
const { creatProcessPool } = require('./pool')

const doWork = creatProcessPool('./pool.worker.js')
doWork('A', function (error, msg) {
    if (error) {
        console.log(error.message)
        return
    }

    console.log('运算结果是:', msg)
})

doWork('B', function (error, msg) {
    if (error) {
        console.log(error.message)
        return
    }

    console.log('运算结果是:', msg)
})

```


### 如何创建和关闭孤儿进程？


> 什么是孤儿进程？


一个父进程退出，而它的一个或多个子进程还在运行，那么那些子进程将成为孤儿进程。孤儿进程将被init进程(进程号为1)所收养，并由init进程对它们完成状态收集工作。


下面代码中，就是主进程将socket通过IPC传递给子进程，然后退出。从而子进程成为孤儿进程，并且将监听8888的socket交给server对象，从而本地访问8888端口，看到对应输出：


```javascript
import * as cp from 'child_process';
import * as http from 'http';
import * as net from 'net';
import { fileURLToPath } from 'url';
import path from 'path';

// package.json 中的 type 设置为 module 后，需要转一下：<https://bobbyhadz.com/blog/javascript-dirname-is-not-defined-in-es-module-scope>
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

orphan();

// 孤儿进程：主进程先退出，ppid变为1
function orphan() {
  if (!process.send) {
    // 主进程中
    const server = net.createServer();
    server.listen(8888);

    const worker = cp.fork(__filename);
    worker.send('server', server); // 把net.Server/net.Socket传递给子进程
    console.log(
      'worker process created, pid: %s ppid: %s',
      worker.pid,
      process.pid,
    );
    process.exit();
  } else {
    const server = http.createServer((req, res) => {
      res.end('I am worker, pid: ' + process.pid + ', ppid: ' + process.ppid); // 记录当前工作进程 pid 及父进程 ppid
    });

    let worker;
    process.on('message', (msg, sendHandle) => {
      if (msg === 'server') {
        worker = sendHandle;
        worker.on('connection', (socket) => {
          server.emit('connection', socket);
        });
      }
    });
  }
}

```


启动后，通过 `ps -o pid,ppid,state,tty,command | grep 'orphan.js'` 查看结果，可以看到33278的ppid已经变成1，托管给了init进程：


![e6c9d24egy1h1h922iqarj21za03gwg4.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-17-process-cluster/e6c9d24egy1h1h922iqarj21za03gwg4.jpg)


> 如何关闭孤儿进程？


通过 `kill -9 [PID]` 就可以关闭孤儿进程。


### 如何创建和关闭僵尸进程？


> 什么是僵尸进程？


一个进程使用fork创建子进程，如果子进程退出，而父进程并没有调用wait或waitpid获取子进程的状态信息，那么子进程的进程描述符仍然保存在系统中。这种进程称之为僵死进程。


> 僵尸进程和孤儿进程区别是？

- 子进程退出了，但是进程描述符仍然存在
- 子进程没有危害，但是僵尸进程会占用进程描述符（有限的系统资源），所以是有害的

下面是创建僵尸进程的例子：


```javascript
import * as cp from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);

zombie();

function zombie() {
  if (process.send) {
    console.log(process.ppid, process.pid);
    return process.exit();
  }

  const child = cp.fork(__filename);
  while (1) {} // 主进程永久阻塞
}

```


在命令行查看


```text
ps -o pid,ppid,state,tty,command | grep 'node'
```


，就能看到创建的子进程状态已经变成 Z+:


![e6c9d24egy1h1h9ny7uq6j20i801k74b.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-17-process-cluster/e6c9d24egy1h1h9ny7uq6j20i801k74b.jpg)


> 如何关闭僵尸进程？


通过直接关闭僵尸进程的父进程，可以关闭僵尸进程。


在NodeJS中，子进程退出后，父进程可以感知到并且清理子进程资源，**正常情况下，开发者无需感知**。前面的代码之所以能成为僵尸进程，因为利用`while(1){}`吃满了父进程的CPU，无法处理子进程的退出信号。下面的代码中，子进程退出后，父进程可以监听到，并且没有僵尸进程产生：


```javascript
function zombie() {
  if (process.send) {
    console.log(process.ppid, process.pid);
    return process.exit();
  }

  const child = cp.fork(__filename);

  child
    .on('exit', () => {
      console.log('exit');
    })
    .on('close', () => {
      console.log('close');
    });
}

```


### 如何创建守护进程？


> 什么是守护进程？


Linux Daemon（守护进程）是运行在后台的一种特殊进程。它独立于控制终端并且周期性地执行某种任务或等待处理某些发生的事件。


> 如何创建守护进程？


根据[nodejs文档](http://nodejs.cn/api/child_process.html#optionsdetached)，最关键的是：

- detached 设置为 true，让子进程在父进程退出后可自己运行
- 调用 `subprocess.unref()` ，不将子进程包括在父进程的引用计数中，从而方便父进程退出
- stdio 设置成 ignore，或者其他IO，将父子进程的IO中断，从而方便父进程退出

参考了 [daemon.js](https://github.com/indexzero/daemon.node/blob/master/index.js) 库的实现：


```javascript
const daemon = function (script, args, opt) {
  opt = opt || {};

  const stdout = opt.stdout || 'ignore';
  const stderr = opt.stderr || 'ignore';

  const env = opt.env || process.env;
  const cwd = opt.cwd || process.cwd();

  const cp_opt = {
    stdio: ['ignore', stdout, stderr], // 子进程的stdin一定要是ignore
    env: env,
    cwd: cwd,
    detached: true,
  };

  // spawn the child using the same node process as ours
  const child = spawn(process.execPath, [script].concat(args), cp_opt);

  // required so the parent can exit
  child.unref();

  return child;
};

daemon('daemon-worker.js', [], {});

```


daemon-worker.js 守护进程逻辑是定时输出到指定文件，代码如下：


```javascript
import { createWriteStream } from 'fs';
import { Console } from 'console';

// custom simple logger
const logger = new Console(
  createWriteStream('./stdout.log'),
  createWriteStream('./stderr.log'),
);

setInterval(function () {
  logger.log('daemon pid: ', process.pid, ', ppid: ', process.ppid);
}, 1000 * 10);

```


运行之后，主进程退出，子进程变成守护进程，交由init进程托管。通过 `ps aux -o pid,ppid,state,tty,command | grep 'daemon-worker.js'` 查看结果：


![e6c9d24egy1h1hc0038xbj228y03egnx.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2022-04-17-process-cluster/e6c9d24egy1h1hc0038xbj228y03egnx.jpg)


> 守护进程和孤儿进程的区别是？


在 [Difference between a Daemon process and an orphan process?](https://stackoverflow.com/questions/42015522/difference-between-a-daemon-process-and-an-orphan-process) 中指出，守护进程是孤儿进程的一种特殊情况，它是由开发者「主动」孤立的，为了让其一直运行某种任务。


在 [Linux 之守护进程、僵死进程与孤儿进程](https://liubigbin.github.io/2016/03/11/Linux-%E4%B9%8B%E5%AE%88%E6%8A%A4%E8%BF%9B%E7%A8%8B%E3%80%81%E5%83%B5%E6%AD%BB%E8%BF%9B%E7%A8%8B%E4%B8%8E%E5%AD%A4%E5%84%BF%E8%BF%9B%E7%A8%8B/) 也提到了unix服务基本上是通过守护进程来启动的，后缀通常是`d`，比如 sshd、crond 等。


## 参考链接

- [nodejs 学习笔记](https://juejin.im/post/5d5158eff265da03e83b60ce)
- [一篇文章构建你的 NodeJS 知识体系](https://juejin.im/post/5c4c0ee8f265da61117aa527)
- [Node.js - 进程学习笔记](https://bennyzheng.github.io/archivers/2016/12/node-process/)
- [glob](https://baike.baidu.com/item/glob/9515871?fr=aladdin)
- [Nodejs 进阶：如何玩转子进程（child_process）](https://www.cnblogs.com/chyingp/p/node-learning-guide-child_process.html)
- [孤儿进程与僵尸进程[总结]](https://www.cnblogs.com/Anker/p/3271773.html)
- [什么是守护进程？](https://www.zhihu.com/question/38609004/answer/77190522)
- [Linux 之守护进程、僵死进程与孤儿进程](https://liubigbin.github.io/2016/03/11/Linux-%E4%B9%8B%E5%AE%88%E6%8A%A4%E8%BF%9B%E7%A8%8B%E3%80%81%E5%83%B5%E6%AD%BB%E8%BF%9B%E7%A8%8B%E4%B8%8E%E5%AD%A4%E5%84%BF%E8%BF%9B%E7%A8%8B/)

