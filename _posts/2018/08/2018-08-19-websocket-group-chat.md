---
title: "原生websocket学习和群聊实现"
date: 2018-08-19
permalink: /2018-08-19-websocket-group-chat/
---
`WebSocket`协议可以实现前后端全双工通信，从而取代浪费资源的长轮询。在此协议的基础上，可以实现前后端数据、多端数据，真正的**实时响应**。在学习`WebSocket`的过程中，实现了一个简化版群聊，过程和代码详细记录在这篇文章中。


## 概述


### WebSocket 是什么？

1. 建立在 TCP 协议之上的网络通信协议
2. 全双工通信协议
3. 没有同源限制
4. 可以发送文本、二进制数据等

### 为什么需要 WebSocket？


了解计算机网络协议的人，应该都知道：HTTP 协议是一种无状态的、无连接的、单向的应用层协议。它采用了请求/响应模型。通信请求只能由客户端发起，服务端对请求做出应答处理。


这种通信模型有一个弊端：HTTP 协议无法实现服务器主动向客户端发起消息。


因此，如果在客户端想实时监听服务器变化，必须使用 ajax 来进行轮询，效率低，浪费资源。


而 websocket 就可以使得**前后端进行全双工通信（两方都可以向对方进行数据推送），是真正的平等对话**。


## WebSocket 客户端


支持`HTML5`的浏览器支持 WebSocket 协议：


```typescript
var ws = new WebSocket(url); // 创建一个websocket对象
```


### WebSocket 属性


![Untitled.png](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2018-08-19-websocket-group-chat/e31120842cb2f0f16e5fb24e4fcbb411.png)


child_database


### WebSocket 方法


child_database


### Websocket 事件


child_database


### 代码实现


假设我们在本地`8080`端口打开了 websocket 服务，那么，下面代码可以在浏览器中实现和这个服务的通信：


```typescript
<body>
    <script>
        var ws = new WebSocket("ws://localhost:8080/");
        // 建立连接触发
        ws.onopen = function() {
            ws.send("open ws");
            console.log("open ws");
        };
        // 接收服务端数据触发
        ws.onmessage = function(evt) {
            var data = evt.data;
            console.log("Data is ", data);
        };
        // 断开连接触发
        ws.onclose = function() {
            console.log("close ws");
        };
    </script>
</body>
```


## WebSocket 服务端


> 关于服务端实现，根据技术选型不同，可以选用不同的库和包。我这里使用的是node的ws库来 websocket 服务端。


在[阮一峰的博文](http://www.ruanyifeng.com/blog/2017/05/websocket.html)提到的`socket.io`库，在浏览器端的写法不兼容原生 API，准确来说，它们自己实现了一套 websocket。所以，使用的时候前后端都应该引用第三方库。**这样就造成了代码迁移性，严重下降。**


综上所述，`ws`库有以下优点：

1. 兼容性好，兼容浏览器原生 API
2. 长期维护，效果稳定
3. 使用方便（往下看就知道了）

## 实现群聊


### 群聊 服务端实现


首先，在命令行中，安装`ws`库: `npm install ws --save`


现在，利用`ws`来实现一个监听`8080`端口的 websocket 服务器，**讲解都在代码注释里，一目了然**：


```typescript
const PORT = 8080; // 监听端口
const WebSocket = require("ws"); // 引入 ws 库
const wss = new WebSocket.Server({ port: PORT }); // 声明wss对象
/**
 * 向除了本身之外所有客户端发送消息，实现群聊功能
 * @param {*} data 要发送的数据
 * @param {*} ws 客户端连接对象
 */
wss.broadcastToElse = function broadcast(data, ws) {
    wss.clients.forEach(function each(client) {
        if (client !== ws && client.readyState === WebSocket.OPEN) {
            client.send(data);
        }
    });
};
/* 客户端接入，触发 connection */
wss.on("connection", function connection(ws, req) {
    let ip = req.connection.remoteAddress; // 通过req对象可以获得客户端信息，比如：ip，headers等
    /* 客户端发送消息，触发 message */
    ws.on("message", function incoming(message) {
        ws.send(message); // 向客户端发送消息
        wss.broadcastToElse(message, ws); // 向 其他的 客户端发送消息，实现群聊效果
    });
});
```


### 群聊 客户端实现


为了方便编写，这里引入了`jquery`和`bootstrap`这两个库，只需要关注 js 代码即可。


```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>群聊</title>
        <link
            href="<https://cdn.bootcss.com/bootstrap/3.3.7/css/bootstrap.min.css>"
            rel="stylesheet"
        />
        <script src="<https://cdn.bootcss.com/jquery/3.3.0/jquery.min.js>"></script>
    </head>
    <body>
        <div class="container">
            <textarea
                class="form-control"
                rows="30"
                disabled="disabled"
                id="show-area"
            ></textarea>
            <input
                type="text"
                class="form-control"
                placeholder="请输入聊天内容"
                id="chat-input"
            />
            <button type="button" class="btn btn-info" id="send-btn">
                发送
            </button>
        </div>
        <script>
            var userName = parseInt(Math.random() * 1000, 10); // 随机用户名, 以标识身份
            var sendBtn = $("#send-btn"), // 发送信息按钮
                chatInput = $("#chat-input"), // 聊天信息输入框
                showArea = $("#show-area"); // 聊天信息展示框
            var ws = new WebSocket("ws://localhost:8080/"); // 初始化WebSocket对象
            sendBtn.on("click", function() {
                var content = chatInput.val();
                if (content.length === 0) {
                    return alert("请不要输入空白内容");
                }
                content =
                    "At " +
                    new Date().toString() +
                    "\\n" +
                    "来自用户" +
                    userName +
                    "\\n" +
                    content; // 拼接用户信息、时间信息和消息
                ws.send(content); // 发送消息
                chatInput.val(""); // 清空输入框
            });
            ws.onopen = function() {
                console.log("Conncet open");
            };
            ws.onmessage = function(evt) {
                var data = evt.data;
                showArea.val(showArea.val() + data + "\\n\\n"); // 刷新聊天信息展示框：显示群聊信息
            };
            ws.onclose = function() {
                console.log("Connect close");
            };
        </script>
    </body>
</html>
```


### 群聊 效果展示


首先启动我们的服务端代码：`node server.js` 。其中，`server.js`是放置服务端代码的文件。


然后，我们打开 2 次编写的`html`代码，这相当于，打开 2 个客户端。来检测群聊功能。


## 相关资料

- 概念解释：
	- [http://www.ruanyifeng.com/blog/2017/05/websocket.html](http://www.ruanyifeng.com/blog/2017/05/websocket.html)
	- [https://www.cnblogs.com/jingmoxukong/p/7755643.html](https://www.cnblogs.com/jingmoxukong/p/7755643.html)
- `ws`文档：[https://www.npmjs.com/package/ws](https://www.npmjs.com/package/ws)

