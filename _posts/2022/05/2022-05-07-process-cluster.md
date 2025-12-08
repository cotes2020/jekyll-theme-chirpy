---
title: "NodeJS 集群：cluster.js 和 PM2"
date: 2022-05-07
permalink: /2022-05-07-process-cluster/
categories: ["实战分享"]
---
Nodejs 提供了 cluster 来支持多进程集群，提高多核 CPU 的利用效率，实现负载均衡，最大程度利用机器性能。


同时，在 cluster 模块之前，社区里也有 pm2 来在本地启动多个nodejs服务进程，并且有完备的日志、监控、重启等策略。


## 进程集群模块 - cluster


从以下几个方面介绍 cluster 的 API 和用法：

- cluster 启动 HTTP 服务器
- 如何进行广播？
- 如何实现状态共享？
- 如何处理进程退出？
- 更多进程控制方法：心跳保活、自动重启、负载检测

### cluster 启动 HTTP 服务器


为了方便测试，全局安装 autocannon：


```shell
npm install -g autocannon
```


不借助 cluster，编写一个简单的 http 服务器：


```javascript
const http = require("http");
http.createServer((req, res) => {
    // 模拟cpu计算
    for (let i = 0; i < 100000; ++i) {}
    res.statusCode = 200;
    res.end("hello world!");
}).listen(4000);

```


> 这里加入了「循环」来消耗CPU。正常开发中，要规避在 NodeJS 主进程中出现 CPU 密集型操作。


借助 autocannon 开启 1000 个连接，每个连接的请求次数为 10 次，压测结果如下：


```text
➜  _posts git:(master) ✗ autocannon -c 1000 -p 10 <http://127.0.0.1:4000>
Running 10s test @ <http://127.0.0.1:4000>
1000 connections with 10 pipelining factor
┌─────────┬──────┬──────┬────────┬────────┬──────────┬───────────┬────────────┐
│ Stat    │ 2.5% │ 50%  │ 97.5%  │ 99%    │ Avg      │ Stdev     │ Max        │
├─────────┼──────┼──────┼────────┼────────┼──────────┼───────────┼────────────┤
│ Latency │ 0 ms │ 0 ms │ 636 ms │ 650 ms │ 62.48 ms │ 197.51 ms │ 2928.64 ms │
└─────────┴──────┴──────┴────────┴────────┴──────────┴───────────┴────────────┘
┌───────────┬─────────┬─────────┬─────────┬─────────┬──────────┬────────┬─────────┐
│ Stat      │ 1%      │ 2.5%    │ 50%     │ 97.5%   │ Avg      │ Stdev  │ Min     │
├───────────┼─────────┼─────────┼─────────┼─────────┼──────────┼────────┼─────────┤
│ Req/Sec   │ 13095   │ 13095   │ 15911   │ 16303   │ 15558.91 │ 901.48 │ 13092   │
├───────────┼─────────┼─────────┼─────────┼─────────┼──────────┼────────┼─────────┤
│ Bytes/Sec │ 1.47 MB │ 1.47 MB │ 1.78 MB │ 1.83 MB │ 1.74 MB  │ 101 kB │ 1.47 MB │
└───────────┴─────────┴─────────┴─────────┴─────────┴──────────┴────────┴─────────┘
Req/Bytes counts sampled once per second.
171k requests in 11.17s, 19.2 MB read
50 errors (0 timeouts)

```


然后用 cluster 模块来启动一个利用多核的 http 服务器，代码如下：


```javascript
const cluster = require("cluster");
const http = require("http");
const os = require("os");
if (cluster.isMaster) {
    const cpuNum = os.cpus().length;
    for (let i = 0; i < cpuNum; ++i) {
        cluster.fork();
    }
} else {
    runServer();
}
function runServer() {
    http.createServer((req, res) => {
        for (let i = 0; i < 100000; ++i) {}
        res.statusCode = 200;
        res.end("hello world!");
    }).listen(4000);
}

```


同样利用 autocannon 进行测试，结果如下：


```text
➜  _posts git:(master) ✗ autocannon -c 1000 -p 10 <http://127.0.0.1:4000>
Running 10s test @ <http://127.0.0.1:4000>
1000 connections with 10 pipelining factor
┌─────────┬──────┬──────┬────────┬────────┬─────────┬──────────┬──────────┐
│ Stat    │ 2.5% │ 50%  │ 97.5%  │ 99%    │ Avg     │ Stdev    │ Max      │
├─────────┼──────┼──────┼────────┼────────┼─────────┼──────────┼──────────┤
│ Latency │ 0 ms │ 0 ms │ 113 ms │ 125 ms │ 11.5 ms │ 37.37 ms │ 807.5 ms │
└─────────┴──────┴──────┴────────┴────────┴─────────┴──────────┴──────────┘
┌───────────┬────────┬────────┬─────────┬─────────┬─────────┬──────────┬────────┐
│ Stat      │ 1%     │ 2.5%   │ 50%     │ 97.5%   │ Avg     │ Stdev    │ Min    │
├───────────┼────────┼────────┼─────────┼─────────┼─────────┼──────────┼────────┤
│ Req/Sec   │ 43711  │ 43711  │ 97023   │ 108671  │ 90811.2 │ 16898.34 │ 43710  │
├───────────┼────────┼────────┼─────────┼─────────┼─────────┼──────────┼────────┤
│ Bytes/Sec │ 4.9 MB │ 4.9 MB │ 10.9 MB │ 12.2 MB │ 10.2 MB │ 1.89 MB  │ 4.9 MB │
└───────────┴────────┴────────┴─────────┴─────────┴─────────┴──────────┴────────┘
Req/Bytes counts sampled once per second.
908k requests in 10.7s, 102 MB read

```


可以看到，错误请求从 50 降低到 0，最长请求延迟从 2.9s 降低到了 0.8s，平均请求量从 1.5w 提升到了 9w，平均下载量从 1.74MB 提升到了 10.2MB。而本机的`os.cpus().length`返回的结果是 12，提升非常稳定，和 cpu 核数基本成正比。


从上面的实践也看到，从 cluster 开启的子进程总数量最好和 cpu 数量一样。


### 如何进行广播？


广播需要父子进程之间进行通信，多用于消息下发、数据共享。cluster 是基于 `child_process` 模块的，所以通信的做法和 `child_process` 区别不大。


在主进程中， `cluster.workders` 是个哈希表，可以遍历得到所有工作进程。如下所示，给所有的工作进程广播消息：


```typescript
if (cluster.isMaster) {
    for (let i = 0; i < os.cpus().length; ++i) {
        cluster.fork();
    }
    // 给工作进程广播消息
    for (const id in cluster.workers) {
        cluster.workers[id].send({
            data: "msg"
        });
    }
} else if (cluster.isWorker) {
    // 工作进程接受到消息
    process.on("message", msg => {
        console.log("msg is", msg);
    });
}

```


### 如何实现状态共享？


在上一个例子中，看到了借助 cluster.workers 和事件机制，来进行消息广播。但由于集群的每个节点是“分散”，所以对于有状态的服务应该想办法解决“状态共享”这个问题。


总体的解决思路：

1. 需要有一个进程专门存放公共状态，当其它进程想更新公共状态时，需要通过 IPC 来「通知」存放公共状态的进程来进行更新。
2. 进程更新公共状态后，需要像其它进程「同步」最新的公共状态。

例如有需要我们进行总访问量统计的需求，并且将当前的访问量返回给客户端。由于每个进程都承载了一部分访问，工作进程接收到请求的时候，需要向主进程上报；工作进程接收到上报，更新访问总量，并且广播给各个工作进程。这就是一个完整的消息上报 => 状态更新 => 消息广播的过程。


按照上面的思路，假设让主进程负责维护公共状态，统计总共的访问次数。代码如下：


```typescript
// 工作进程逻辑
function runServer() {
    let visitTotal = 0;
    // 接收主进程的广播
    process.on("message", msg => {
        if (msg.tag === "broadcast") visitTotal = msg.visitTotal;
    });
    http.createServer((req, res) => {
        // 消息上报给主进程
        process.send({
            tag: "report"
        });
        res.statusCode = 200;
        res.end(`visit total times is ${visitTotal + 1}`);
    }).listen(4000);
}

```


是的，就是通过传递消息上的一个字段，来标识是工作进程上报的消息还是主进程广播的消息。给主进程用的 broadcast() 函数如下：


```typescript
function broadcast(workers, data) {
    for (const id in workers) {
        // 给工作进程广播消息
        workers[id].send({
            tag: "broadcast",
            ...data
        });
    }
}

```


最后，主进程中需要为工作进程添加`message`事件的监听器，这样才能收到工作进程的消息，并且更新保存在主进程中的状态（visitTotal），完成广播。代码如下：


```typescript
if (cluster.isMaster) {
    const cpuNum = os.cpus().length;
    for (let i = 0; i < cpuNum; ++i) {
        cluster.fork();
    }
    listenWorker()
} else if (cluster.isWorker) {
    runWorker();
}

// 监听来自工作进程的消息
function listenWorker(initVisitTotal = 0) {
    let visitTotal = initVisitTotal || 0;

    for (const id in cluster.workers) {
        cluster.workers[id].on("message", msg => {
            // 如果是report类型的消息：
            // 1. 更新总访问数
            // 2. 给所有的工作进程同步
            if (msg.tag === "report") {
                ++visitTotal;
                broadcast(cluster.workers, { visitTotal });
            }
        });
    }
}

// 主进程给工作进程「广播」
function broadcast(workers, data) {
    for (const id in workers) {
        workers[id].send({
            tag: "broadcast",
            ...data
        });
    }
}

// 1. 工作进程监听来自主进程的广播，同步公共状态（访问次数）
// 2. 请求每次进来，给主进程上报，由主进程来更新公共状态。
function runWorker() {
    let visitTotal = 0;
    // 接收主进程的广播
    process.on("message", msg => {
        if (msg.tag === "broadcast") visitTotal = msg.visitTotal;
    });
    http.createServer((req, res) => {
        // 消息上报给主进程
        process.send({
            tag: "report"
        });
        res.statusCode = 200;
        res.end(`visit total times is ${visitTotal + 1}`);
    }).listen(4000);
}

```


> 更常用的做法是专门准备一个服务器来进行统计，将服务单独部署。这里是为了深入理解和学习 cluster 模块举的例子。


### 如何处理工作进程退出？


cluster 模块中有 2 个 exit 事件：一个是 Worker 上的，仅用于工作进程中；另一个是主进程上，任何一个工作进程关闭都会触发。


在工作进程正常退出的时候，code 为 0，并且 Worker 上的 exitedAfterDisconnect 属性为 true。那么检测 code 和 exitedAfterDisconnect 属性，就能判断进程是否是异常退出。并且重新 fork 一个新的工作进程，来保持服务稳定运行。代码如下：


```typescript
cluster.on("exit", (worker, code, signal) => {
    if (code || !worker.exitedAfterDisconnect) {
        console.log(`${worker.id} 崩溃，重启新的子进程`);
        cluster.fork();
    }
});

```


注意，exitedAfterDisconnect 属性在正常退出、调用 worker.kill() 或调用 worker.disconnect()时，均被设置为 true。因为调用 kill 和 disconnect 均为代码逻辑主动执行，属于程序的一部分。


### 调度多进程细节：心跳保活、自动重启、负载检测


除了前面所讲的方法，进程控制的常见方法还有：心跳保活、自动重启、负载检测。


**心跳保活**：工作进程定时向主进程发送心跳包，主进程如果检测到长时间没有收到心跳包，要关闭对应的工作进程，并重启新的进程。


**自动重启**：给每个工作进程设置一个“生命周期”，例如 60mins。到时间后，通知主进程进行重启。


**负载检测**：工作进程和主进程可以定期检测 cpu 占用率、内存占用率、平均负载等指标，过高的话，则关闭重启对应工作进程。关于检测方法可以看这篇文章[《NodeJS 模块研究 - os》](https://0x98k.com/2020-01-11-nodejs-os/)。


这些方法在 vemojs 中都有应用，具体可以看这篇文章：[《VemoJS 源码拆解》](https://dongyuanxin.github.io/2019-04-23-vemojs/)


## 多进程管理工具 - pm2


在NodeJS中，pm2 是最常用的多进程管理工具。他提供了各种命令，并且具备强大的监控、日志、重启、rpc等功能。


在实际生产中，pm2 比 `cluster` 集群更常用。


### 常见命令


基本的命令，请见：[pm2 quick start](https://pm2.keymetrics.io/docs/usage/quick-start/)


一些不常用但重要的命令：

- 如何传递参数？`pm2 start app.js -- arg1 arg2`
- 如何启动配置文件？`pm2 start ecosystem.config.js`
- 如何生成配置文件模版？`pm2 init simple`
- 如何查看交互式shell控制面板（包括日志）？`pm2 monit`

### 重启策略

- **定时重启：**
	- 命令：`-cron-restart="0 0 * * *"`
	- 用途：koa/egg/nest服务器，长期间运行容易出现内存一直过高占用。此时可以设置每天凌晨定时重启。
- 基于内存
	- 命令：`-max-memory-restart 300M`
	- 用途：防止内存泄漏等导致的内存占用过高的情况
- 禁用自动重启：
	- 命令：`-no-autorestart`
	- 用途：一次性脚本，无须像服务器那样保活。
- **根据文件变动监听重启：**
	- 配置：`watch`(监听的文件夹)、`watch_delay`(监听防抖延时)、`ignore_watch`(忽略的文件夹)
	- 用途：线上服务如果部署在K8S下，配置文件是中心下发的，那么就可以监听配置文件变动；本地开发也会用到。
- **避免连续重启：**
	- 命令：`-exp-backoff-restart-delay=100`
	- 用途：避免在指定时间内，不断重启程序，导致第三方服务（数据库、消息队列）被不断连接，压力过大。

### 开机启动


通过 `pm2 startup` 可以得到，在当前机器上，将 pm2 设置为开机自动启动的脚本。不过现在都用k8s+docker，用处不大。


通过 `pm2 save` 可以将当前应用列表进行「快照」，方便 `pm2 startup` 启动。


### 利用多核性能


pm2 支持 cluster 模式。通过配置文件的 `exec_mode : "cluster"` 指定。


通过 `instances` 配置，可以指定启动工作进程的数量。


在k8s中，workload上的单个pod，如果是1核以上的配置，那么就可以使用pm2来启动多个工作进程，从而利用CPU。


不过我们在实践中，pod都是单核的，相当于调度交给了k8s，而不是pm2，所以很少用到pm2。


### 优雅启动/关闭


在 [pm2 start and end](https://pm2.keymetrics.io/docs/usage/signals-clean-restart/)中提到了一些优雅启动和关闭的情况。


当服务启动时，需要连接redis、mongodb等第三方中间件。未连接前，服务状态不能是 online 。实现方法：

- 配置打开 `wait_ready` ，设置为 true
- 当第三方中间件均连接完成后，发送 `ready` 消息：`process.send('ready')`
- 启动超时的时间，可以通过 `listen_timeout` 来设置

当服务关闭时，也需要关闭redis、mongodb等第三方中间件。否则容易造成连接未关闭，但是服务退出了。处理方法：

- 监听 `SIGINT` 信号量。然后在回调里，关闭各个连接后，再调用 `process.exit([code])` 主动退出
- 退出超时时间，可以通过 `kill_timeout` 来设置。

> 进程退出时，pm2 是如何给进程发送信号量的？

1. 首先发送 `SIGINT` 信号量
2. 如果进程在指定时间退出了，那么就结束。否则，pm2 会再次发起 `SIGKILL` 信号量强制退出。

可以通过设置环境变量 `PM2_KILL_SIGNAL` 将信号 `SIGINT` 替换为任何其他信号（例如`SIGTERM`） 。


## 参考链接

- [Nodejs 文档](http://nodejs.cn/api/cluster.html)
- [NodeJS 模块研究 - os](https://dongyuanxin.github.io/2020-01-11-nodejs-os/)
- [autocannon](https://www.npmjs.com/package/autocannon)
- [解读 NodeJS 的 Cluster 模块](https://www.notion.so/0x98k/alloyteam.com/2015/08/nodejs-cluster-tutorial/)
- [Node.js 集群（cluster）：扩展你的 Node.js 应用](https://zhuanlan.zhihu.com/p/36728299)
- [pm2 docs](https://pm2.keymetrics.io/docs/usage/quick-start/)

