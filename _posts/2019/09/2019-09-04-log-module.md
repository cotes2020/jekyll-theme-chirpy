---
title: "NodeJS日志库的设计与优化"
url: "2019-09-04-log-module"
date: 2019-09-04
---

Keywords：堆栈、容器存储、Lazy Log、异步日志、缓存周期


## 概述


规范化的日志输出和存留，可以用来：**开发调试、行为留存、程序状态记录**。


对于日志，一般需要 4 个要素：**时间、级别、位置、内容、上下文信息**。对于集群或者多台机器来说，日志还需要区分不同机器的**唯一标识**。


## 基本原理：堆栈信息


自己封了个包，日志报错信息的格式为：`<time> <level> <loc> <info>`。下面就是一条日志：


```shell
[1:17:27 PM] <warning> (/root/tcb-console-node/dist/controllers/auth.js:auth:38:15) smart-proxy signature check fail
```


time、level、info 元素非常容易获得，但是对于 loc 元素来说，它包含了调用日志的文件、函数、行数和列数。这些是通过**堆栈信息**来获得的。


所以，获取 loc 的原理是：调用日志模块接口时，接口内部生成一个 Error；根据堆栈信息，按照规范撰写正则表达式，匹配出文件、函数、行数和列数。


```typescript
// Node Error Stack: <https://github.com/v8/v8/wiki/Stack%20Trace%20API>
const stackReg1 = /at\\s+(.*)\\s+\\((.*):(\\d*):(\\d*)\\)/i;
const stackReg2 = /at\\s+()(.*):(\\d*):(\\d*)/i;
export function print(msg: string, level?: LogLevel) {
    const time = new Date().toLocaleTimeString();
    const error = new Error(); // 主动生成错误
    const stackList = error.stack.split("\\n").slice(2); // 处理报错堆栈
    const sp = stackReg1.exec(stackList[0]) || stackReg2.exec(stackList[0]); // 从堆栈中匹配信息
    if (!sp) {
        return;
    }
    const log = {
        time,
        func: sp[1],
        filepath: sp[2],
        line: sp[3],
        pos: sp[4],
        stack: error.stack,
        msg,
        level
    };
    // ...
}
```


⚠️ 注意：在`error.stack.split('\\n').slice(2)`这句逻辑中，对于不同的调用层级关系，切片的位置不一样。上面暴露的 print 函数，外界是直接调用。如果是外界调用的接口 a，接口 a 调用 b，接口 b 中生成的 Error。那么，堆栈会变长。但根据 Nodejs 的文档，堆栈最多是 10 层。


## 日志存储


日志可以根据级别，写入指定文件。比如: info 级别 => /data/my-logs/info.log。


程序应该自动识别环境，开发环境下，可以只吐到控制台，无需写入磁盘。


## 优化方法


### 1. Lazy Log


主要体现：**根据不同环境、不同级别中，节省 IO**。


对于开发环境，日志直接输出控制台即可，没必要向磁盘写入。


对于 log、info 等日志级别，日志直接输出控制台，开发/生产环境均没必要向磁盘写入。


### 2. 异步打印日志


对于高并发服务，每次均向控制台/磁盘采用**同时策略**吐出日志，会造成 IO 过高。


可以自己封装个方法，将日志存放在队列中，每隔 1000ms 打印/磁盘 io 一次，再清空队列。


```typescript
let queue = [];
let lock = false;
const interval = 1000;
setTimeout(() => {
    if (!queue.length || lock) return;
    lock = true; // 根据实际情况，决定是否用锁
    let copyQueue = queue;
    queue = []; // 申请新的内存空间
    copyQueue.forEach(item => console.log(item)); // 控制台/磁盘io
    copyQueue = null; // 放置内存泄漏
    lock = false;
}, interval);
export function print(msg: string, level?: LogLevel) {
    const error = new Error(); // 主动生成错误
    const stackList = error.stack.split("\\n").slice(2); // 处理报错堆栈
    const sp = stackReg1.exec(stackList[0]) || stackReg2.exec(stackList[0]); // 从堆栈中匹配信息
    if (!sp) return;
    queue.push({
        time: new Date().toLocaleTimeString(),
        func: sp[1],
        filepath: sp[2],
        line: sp[3],
        pos: sp[4],
        stack: error.stack,
        msg,
        level
    });
}
```


⚠️ 可以使用**消息队列**。云厂商的日志服务就是这个思路，开启脚本监听对应日志文件，异步将数据放上云端。


### 3. 缓存周期


对于程序日志来说，可以设置 15 天自动清理。对于敏感接口访问留存，可以持久存储在 DB 中。


### 4. ELK


用于日志可视化，以及日志快捷查询。


## 成熟的库

- [tracer](https://www.npmjs.com/package/tracer)
- [维基百科:消息队列](https://zh.wikipedia.org/zh-hans/%E6%B6%88%E6%81%AF%E9%98%9F%E5%88%97)
- [更多](https://cnodejs.org/topic/5017e156f767cc9a517869b1)

