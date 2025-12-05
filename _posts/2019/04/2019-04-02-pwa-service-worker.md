---
title: "Service Worker离线缓存实战"
url: "2019-04-02-pwa-service-worker"
date: 2019-04-02
---

## 背景介绍


最近实战了 Service Worker（以下简称“sw”）来进行网站缓存，以实现离线状态下，网站仍然可以正常使用。


尤其对于个人博客这种以内容为主体的静态网站，离线访问和缓存优化尤其重要；并且 Ajax 交互较少，离线访问和缓存优化的实现壁垒因此较低。


## 环境准备


虽然 sw 要求必须在 https 环境下才可以使用，但是为了方便开发者，通过`localhost`或者`127.0.0.1`也可以正常加载和使用。


利用 cnpm 下载`http-server`：`npm install http-server -g`


进入存放示例代码的文件目录，启动静态服务器：`http-server -p 80`


最后，准备下 html 代码：


```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>Document</title>
    </head>
    <body>
        <img src="./image.png" height="300" width="300" />
        <img
            src="<https://user-gold-cdn.xitu.io/2017/10/4/50e8f96bbcb3bc644a083a409ce0ce2d?imageView2/0/w/1280/h/960/format/webp/ignore-error/1>"
        />
        <h3>一些提示信息sdfsf</h3>
        <ul>
            <li>浏览器是否支持：<span id="isSupport"></span></li>
            <li>service worker是否注册成功：<span id="isSuccess"></span></li>
            <li>当前注册状态：<span id="state"></span></li>
            <li>当前service worker状态：<span id="swState"></span></li>
        </ul>
        <script src="/script.js"></script>
    </body>
</html>
```


## 注册 Service Worker


我们通过`script.js`来判断浏览器是否支持 serviceWorker，并且加载对应的代码。`script.js`内容如下：


```typescript
window.addEventListener("load", event => {
    // 判断浏览器是否支持
    if ("serviceWorker" in navigator) {
        console.log("支持");
        window.navigator.serviceWorker
            .register("/sw.js", {
                scope: "/"
            })
            .then(registration => {
                console.log("注册成功");
            })
            .catch(error => {
                console.log("注册失败", error.message);
            });
    } else {
        console.log("不支持");
    }
});
```


### 注册时机


如上所示，最好在页面资源加载完成的事件(`window.onload`)之后注册 serviceWorker 线程。**因为 serviceWorker 也会浪费资源和网络 IO**，不能因为它而影响正常情况下（网络信号 ok 的情况）的使用体验。


### 拦截作用域


之后，我们需要用 serviceWorker 线程来拦截资源请求，但不是所有的资源都能被拦截，**这主要是看 serviceWorker 的作用域：它只管理其路由和子路由下的资源文件**。


例如上面代码中，`/sw.js`是 serviceWorker 脚本，它拦截根路径下的所有静态资源。如果是`/static/sw.js`，就只拦截`/static/`下的静态资源。


开发者也可以通过传递`scope`参数，来指定作用域。


## Service Worker 最佳实践


笔者爬了很久的坑，中途看了很多人的博客，包括张鑫旭老师的文章。但是实践的时候都出现了问题，直到读到了百度团队的文章才豁然开朗。


为了让`sw.js`的逻辑更清晰，这里仅仅展示最后总结出来的最优代码。如果想了解更多，可以跳到本章最后一个部分《参考链接》。


### sw 的生命周期


对于 sw，它的生命周期有 3 个部分组成：install -> waiting -> activate。开发者常监听的生命周期是 install 和 activate。


这里需要注意的是：两个事件的回调监听函数的参数上都有`waitUntil`函数。**开发者传递到它的****`promise`****可以让浏览器了解什么时候此状态完成**。


如果难理解，可以看下面这段代码：


```typescript
const VERSION = "v1";
self.addEventListener("install", event => {
    // ServiceWoker注册后，立即添加缓存文件，
    // 当缓存文件被添加完后，才从install -> waiting
    event.waitUntil(
        caches.open(VERSION).then(cache => {
            return cache.addAll(["./index.html", "./image.png"]);
        })
    );
});
```


### 更新 Service Worker 代码


对于缓存的更新，可以通过定义版本号的方式来标识，例如上方代码中的 VERSION 变量。但对于 ServiceWorker 本身的代码更新，需要别的机制。


简单来说，分为以下两步：

1. 在 install 阶段，调用 `self.skipWaiting()` 跳过 waiting 阶段，直接进入 activate 阶段
2. 在 activate 阶段，调用 `self.clients.claim()` 更新客户端 ServiceWorker

代码如下：


```typescript
const VERSION = "v1";
// 添加缓存
self.addEventListener("install", event => {
    // 跳过 waiting 状态，然后会直接进入 activate 阶段
    event.waitUntil(self.skipWaiting());
});
// 缓存更新
self.addEventListener("activate", event => {
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all([
                // 更新所有客户端 Service Worker
                self.clients.claim(),
                // 清理旧版本
                cacheNames.map(cacheName => {
                    // 如果当前版本和缓存版本不一样
                    if (cacheName !== VERSION) {
                        return caches.delete(cacheName);
                    }
                })
            ]);
        })
    );
});
```


### 再探更新


上一部分说了更新 sw 的 2 个步骤，但是为什么这么做呢？


因为对于同一个 sw.js 文件，浏览器可以检测到它已经更新（假设旧代码是 sw1，新代码是 sw2）。由于 sw1 还在运行，以及默认只运行一个同名的 sw 代码，所以 sw2 处于 waiting 状态。**所以需要强制跳过 waiting 状态** 。


进入 activate 后，还需要取得“控制权”，并且弃用旧代码 sw1。上方的代码顺便清理了旧版本的缓存。


### 资源拦截


在代码的最后，需要监听 `fetch`  事件，并且进行拦截。如果命中，返回缓存；如果未命中，放通请求，并且将请求后的资源缓存下来。


代码如下：


```typescript
self.addEventListener("fetch", event => {
    event.respondWith(
        caches.match(event.request).then(response => {
            // 如果 Service Workder 有自己的返回
            if (response) {
                return response;
            }
            let request = event.request.clone();
            return fetch(request).then(httpRes => {
                // http请求的返回已被抓到，可以处置了。
                // 请求失败了，直接返回失败的结果就好了。。
                if (!httpRes || httpRes.status !== 200) {
                    return httpRes;
                }
                // 请求成功的话，将请求缓存起来。
                let responseClone = httpRes.clone();
                caches.open(VERSION).then(cache => {
                    cache.put(event.request, responseClone);
                });
                return httpRes;
            });
        })
    );
});
```


## 效果测试


启动服务后，进入 `localhost` ，打开 devtools 面板。可以看到资源都通过 ServiceWorker


缓存加载进来了。


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554261787790-8516ca44-1872-4e8d-b063-25dab02682b7.png#align=left&display=inline&height=364&name=image.png&originHeight=455&originWidth=1608&size=81057&status=done&width=1286)


现在，我们打开离线模式，


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554261882352-6ef567ff-b6c7-4916-aa5c-89fbbfc9d68f.png#align=left&display=inline&height=520&name=image.png&originHeight=650&originWidth=907&size=62316&status=done&width=726)


离线模式下照样可以访问：


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554261936715-57129714-6312-4e72-8679-7563ff529b83.png#align=left&display=inline&height=725&name=image.png&originHeight=906&originWidth=1920&size=401854&status=done&width=1536)


最后，我们修改一下 html 的代码，并且更新一下 sw.js 中标识缓存版本的变量


VERSION：


![name=image.png](https://cdn.nlark.com/yuque/0/2019/png/233327/1554262033555-b36bfb5a-16ee-4079-a400-b2239a93ee9c.png#align=left&display=inline&height=733&name=image.png&originHeight=916&originWidth=1920&size=285955&status=done&width=1536)


在第 2 次刷新后，通过上图可以看到，缓存版本内容已更新到 v2，并且左侧内容区已经被改变。


## 参考链接

- [本文全部代码地址](https://github.com/dongyuanxin/pwa-service-worker)
- [Service Worker 生命周期](https://developers.google.com/web/fundamentals/primers/service-workers/lifecycle?hl=zh-cn)
- [百度团队：怎么使用 ServiceWorker](https://lavas.baidu.com/pwa/offline-and-cache-loading/service-worker/how-to-use-service-worker)
- [Web Worker 开发模式](https://www.villainhr.com/page/2016/08/22/Web%20Worker)

