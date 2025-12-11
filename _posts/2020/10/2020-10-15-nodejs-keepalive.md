---
title: "Node.js下Keep-Alive的ECONNRESET 问题"
date: 2020-10-15
permalink: /2020-10-15-nodejs-keepalive/
categories: ["C工作实践分享"]
tags: [HTTP协议]
---

## ECONNRESET 问题


### 出现原因


一段长连接，静默一段时间后，服务端没收到数据想主动关闭了 socket。


此时，客户端不知道服务端要关闭 socket，就发送了一个新的 http 请求。


当客户端发完 http 请求，到抵达服务端之间的这段时间，服务端刚好关闭了 socket（客户端才能收到 socket 连接关闭的事件，并触发相关回调逻辑）。


所以当请求抵达服务端，因为服务端关闭了此 socket，自然报错了。


### 解决方法


方法 1: 客户端的 keepalive timeout 小于服务端的 keepalive timeout。


方法 2: 客户端识别错误码，如果错误码是`ECONNRESET`，并且是复用的 socket 连接（keepalive 本质是复用 socket），那么就不能确认是否是错误。**再次发起一次完全一样的请求**。如果正常返回，那么说明是前面说的中间态问题；如果有异常，那么说明是服务端其它问题导致的 socket 关闭。


方法 2 的代码示例：


```typescript
const http = require("http");

const reqInfo = request.get("<http://127.0.0.1:8080>", { agent }, (err) => {
    if (!err) {
        console.log("success");
    } else if (err.code === "ECONNRESET" && reqInfo.req.reusedSocket) {
        return request.get("<http://127.0.0.1:8080>", (err) => {
            if (err) {
                throw err;
            } else {
                console.log("success with retry");
            }
        });
    } else {
        throw err;
    }
});
```


## 事件重复监听


具体代码详见：[Making several http requests using keep-alive mode generates this trace: Warning: Possible EventEmitter memory leak detected.](https://github.com/nodejs/node/issues/9268)


**问题产生的原因**：keepalive 产生的 socket 连接是复用的，所以每次都开启的事件监听，例如 onError，onData，会造成事件重复监听。


**解决思路**：在其它与事件监听相关的场景下也会遇到，可以通过`emitter.setMaxListeners(n)`限制事件数量；或者通过`.once`方法调用，触发后自动卸载事件。


**注意**：新版本 nodejs 无需关心这个问题，已经在 node 代码中修复。


## 参考连接

- [如何解决 Keep-Alive 导致 ECONNRESET 的问题](https://zhuanlan.zhihu.com/p/86953757)
- [node issue](https://github.com/nodejs/node/issues/9268)

