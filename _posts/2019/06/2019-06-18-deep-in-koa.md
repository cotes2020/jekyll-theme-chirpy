---
title: "深入 koa.js 源码：架构设计"
date: 2019-06-18
permalink: /2019-06-18-deep-in-koa/
categories: ["源码精读", "KoaJS"]
---

> 最近读了 koa 的源码，理清楚了架构设计与用到的第三方库。本系列将分为 3 篇，分别介绍 koa 的架构设计和 3 个核心库，最终会手动实现一个简易的 koa。

koa 的实现都在仓库的`lib`目录下，如下图所示，只有 4 个文件：

![](https://static.godbmw.com/img/2019-06-18-deep-in-koa/1.png)

对于这四个文件，根据用途和封装逻辑，可以分为 3 类：req 和 res，上下文以及 application。

## req 和 res

对应的文件是：`request.js` 和 `response.js`。分别代表着客户端请求信息和服务端返回信息。

这两个文件在实现逻辑上完全一致。对外暴露都是一个对象，对象上的属性都使用了`getter`或`setter`来实现读写控制。

## 上下文

对应的文件是：`context.js`。存了运行环境的上下文信息，例如`cookies`。

除此之外，因为`request`和`response`都属于上下文信息，所以通过`delegate.js`库来实现了对`request.js`和`response.js`上所有属性的代理。例如以下代码：

```javascript
/**
 * Response delegation.
 */
delegate(proto, "response")
  .method("attachment")
  .method("redirect");

/**
 * Request delegation.
 */

delegate(proto, "request")
  .method("acceptsLanguages")
  .method("acceptsEncodings");
```

使用代理的另外一个好处就是：更方便的访问 req 和 res 上的属性。比如在开发 koa 应用的时候，可以通过`ctx.headers`来读取客户端请求的头部信息，不需要写成`ctx.res.headers`了（这样写没错）。

**注意**：req 和 res 并不是在`context.js`中被绑定到上下文的，而是在`application`被绑定到上下文变量`ctx`中的。原因是因为每个请求的 req/res 都不是相同的。

## Application

对应的文件是: `application.js`。这个文件的逻辑是最重要的，它的作用主要是：

- 给用户暴露服务启动接口
- 针对每个请求，生成新的上下文
- 处理中间件，将其串联

### 对外暴露接口

使用 koa 时候，我们常通过`listen`或者`callback`来启动服务器：

```javascript
const app = new Koa();
app.listen(3000); // listen启动
http.createServer(app.callback()).listen(3000); // callback启动
```

这两种启动方法是完全等价的。因为`listen`方法内部，就调用了`callback`，并且将它传给`http.createServer`。接着看一下`callback`这个方法主要做了什么：

1. 调用`koa-compose`将中间件串联起来（下文再讲）。
2. 生成传给`http.createServer()`的函数，并且返回。

- `http.createServer`传给函数参数的请求信息和返回信息，都被这个函数拿到了。并且传给`createContext`方法，生成本次请求的上下文。
- 将生成的上下文传给第 1 步生成的中间件调用链，**这就是为什么我们在中间件处理逻辑的时候能够访问`ctx`**

### 生成新的上下文

这里上下文的方法对应的是`createContext`方法。这里我觉得更像语法糖，是为了让 koa 使用者使用更方便。比如以下这段代码：

```javascript
// this.request 是 request.js 暴露出来的对象，将其引用保存在context.request中
// 用户可以直接通过 ctx.属性名 来访问对应属性
const request = (context.request = Object.create(this.request));

// 这个req是本次请求信息，是由 http.createServer 传递给回调函数的
context.req = request.req = response.req = req;
```

读到这里，虽然可以解释 `context.headers` 是 `context.request.headers` 的语法糖这类问题。但是感觉怪怪的。就以这个例子，context.headers 访问的是 context.request 上的 headers，而不是本次请求信息上的`headers`。本次请求信息挂在了`context.req`上。

让我们再回到`reqeust.js`的源码，看到了`headers`的 getter 实现：

```javascript
get headers() {
  return this.req.headers;
}
```

所以，`context.request.headers` 就是 `context.request.req.headers`。而前面提及的`createContext`方法中的逻辑，`context.reqest`上的`req`属性就是由`http`模块函数传来的真实请求信息。 **感谢 [@theniceangel](https://github.com/theniceangel) 的评论指正**。

可以看到，koa 为了让开发者使用方便，在上下文上做了很多工作。

### 中间件机制

中间件的设计是 koa 最重要的部分，实现上用到了`koa-compose`库来串联中间件，形成“洋葱模型”。关于这个库，放在第二篇关于 koa 核心库的介绍中说明。

application 中处理中间件的函数是`use`和`handleRequest`：

- `use`函数：传入`async/await`函数，并将其放入 application 实例上的`middleware`数组中。如果传入是 generator，会调用`koa-conver`库将其转化为`async/await`函数。
- `handleRequest(ctx, fnMiddleware)`函数：传入的`fnMiddleware`是已经串联好的中间件，函数所做的工作就是再其后再添加一个返回给客户端的函数和错误处理函数。返回给客户端的函数其实就是`respond`函数，里面通过调用`res.end()`来向客户端返回信息，整个流程就走完了。
