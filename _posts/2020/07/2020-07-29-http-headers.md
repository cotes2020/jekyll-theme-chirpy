---
title: "HTTP Headers判断请求来源"
date: 2020-07-29
permalink: /2020-07-29-http-headers/
---
## X-Powered-By


`X-`开头的自定义字段。由语言解析器或者应用程序框架输出的，例如 PHP 的输出是：X-Powered-By: PHP/5.2.1


它可以告诉开发者后端使用的是什么框架/语言，所以有安全风险。对于自带此字段的框架/语言，需要手动去掉。


在`express` 或者 `nestjs`中，可以使用：


```typescript
app.disable('x-powered-by');
```


## X-Forwarded-For


`X-`开头的自定义字段。用来识别通过代理或者负载均衡连接到服务端的真实 IP 地址路径。


例如对于 `X-Forwarded-For: client1, proxy1, proxy2` 来说，请求链路如下：

- 从 client1 发出，抵达 proxy1
- proxy1 将 client1 放入 X-Forwarded-For，请求转发到 proxy2
- proxy2 将 proxy1 放入 X-Forwarded-For，请求转发到服务端
- 服务端将 proxy2 放入 X-Forwarded-For，请求不再被转发，返回数据

**可以看到，代理服务器每成功收到一个请求，就把请求来源 IP 地址添加到右边**。


## X-Real-IP


`X-`开头的自定义字段，不属于任何任何标准。用来表示与 HTTP 代理产生 TCP 连接的设备 IP。


**安全问题**


攻击者可以通过手动构造 `X-Real-IP` 和 `X-Forwarded-For` 来伪造真实的源 IP。


解决方法：前面使用 Nginx 代理。


原理：Nginx 会从 TCP 链接中获取真正的 IP 信息，将其追加到`X-Forwarded-For`最右侧，覆盖客户端伪造的`X-Real-IP` 。


```text
location / {
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```


## 参考链接

- [Wiki: X-Forwarded-For](https://zh.wikipedia.org/wiki/X-Forwarded-For)
- [HTTP 请求头中的 X-Forwarded-For](https://imququ.com/post/x-forwarded-for-header-in-http.html)

