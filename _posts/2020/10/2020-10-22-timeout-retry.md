---
title: "超时重试机制"
date: 2020-10-22
permalink: /2020-10-22-timeout-retry/
tags: [重试设计]
---

## 为什么需要超时重试？


如果不设置超时，则会导致请求响应满，**积累导致应用雪崩**。


对于重试来说，是否重试、重试次数以及超时时间，都很重要：

- 是否重试：读请求适合重试，写请求不适合
- 重试次数：一般是 1 次，多了就相当于是 DDos
- 超时时间：内网一般是 2s

### nodejs 相关开发经验


在开发 nodejs 服务的时候，经常需要对数据库的 js 库、redis的js库，进行主动地超时关闭。


并且给出日志和告警，通知开发者。同时，会进行一些自动的重连尝试，尽量做到「自动恢复」。


## 为什么需要主动地进行超时关闭？


1、有些第三方库不支持超时关闭


2、有些第三方库支持 timeout 参数，但是超时并没有关（比如 redis.js，当时就踩过坑）


3、服务主动兜底，不信任第三方库，使服务更健壮


4、主动关闭后，可以根据真实业务需要，来决定是否重试，以及重试的次数时间


## 如何实现主动超时关闭？


观察 axios.js 的源码，发现主动关闭超时连接其实就是调用了`setTimeout`，在指定的超时时间过后，如果还没连接上，就调用 adapter 上的方法销毁释放 http 连接请求，并且触发错误回调。


> **什么是 Axios.js 的 adapter？**  
> axios 支持 ssr、nodejs 以及浏览器这三个环境。adapter 就是 axios 对这三个环境下的原生异步请求接口的一层封装。例如，在 nodejs 环境中，adapter 就是 http 和 https 库。  
> 同样的做法，在 tcb-js-sdk 中也有用到。因为要支持不同的 WebView，例如小程序、web 端、eletron。  
> 本质上，这就是 [JavaScript「结构型」设计模式](https://www.notion.so/1944ae57f02846798b5eefc13baad2d5) 中提到的「代理模式」。


除此之外还要防止错误回调参数被多次调用，以及建立连接后，setTimeout 回调中不可以销毁释放 http 连接资源。


写了个 demo：


```typescript
const http = require("http");

httpClient(
    {
        hostname: "nodejs.cn",
        port: 80,
        path: "/upload",
        method: "GET",
    },
    10,
    (err) => {
        console.log("err.message is", err.message);
    }
);

function httpClient(options, timeout, callback) {
    let callbackRunned = false; // 错误回调只能触发一次

    // 发送请求
    const req = http
        .request(options, (res) => {
            res.on("data", (data) => {
                console.log("data is", data.toString("utf8"));
            });
        })
        .on("connect", () => {
            req.connected = true;
        })
        .on("error", (err) => {
            if (typeof callback === "function" && !callbackRunned) {
                callback(err);
            }
        })
        .end();

    // 超时手动关闭
    setTimeout(() => {
        if (req.connected) {
            // 如果已连接上，就不设置超时
            return;
        }

        if (typeof callback === "function" && !callbackRunned) {
            callbackRunned = true;
            callback(new Error("超时，手动关闭"));
        }
        req.destroy();
    }, timeout || 2000);
}
```


