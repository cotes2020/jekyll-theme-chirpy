---
title: "JS SDK 微内核设计实战"
date: 2020-05-04
permalink: /2020-05-04-microkernel-plugin/
---
## 认识微服务架构


定义：微内核架构（Microkernel Architecture），也被称为插件化架构（Plug-in Architecture），是一种面向功能进行拆分的可扩展性架构，通常用于实现基于产品的应用。


微服务架构内部包括核心系统和插件模块：

- 核心系统：负责与业务功能无关的通用功能，例如插件通信、插件管理等
- 插件系统：负责某个具体业务功能，例如图像检测、短信登录等

核心系统比较稳定，改动低；插件系统改动频繁，能够快速扩展。


## 核心系统设计关键


在[《微内核架构》](https://juejin.im/post/5cbdbeb95188250a8c22abb0)中提到，核心系统的设计关键主要体现在“处理插件”上：

- 插件管理：注册插件的 api、加载插件的时机等
- 插件连接：插件接入核心系统的方式，能够调用核心系统的一些 api
- 插件通信：插件间相互调用

## 实战：tcb-admin-node


上面说了这么多，还是配合代码才能更好理解这种设计的好处。以 tcb-admin-node.js 为例，它不完全是基于微内核的思路进行设计的，但是在扩展能力这块的设计，和微内核思想有着异曲同工之妙。


打开项目根目录下的`index.js`文件，可以看到挂在 Tcb 原型链上的 registerExtension 和 invokeExtension


方法，如下图所示：


![007S8ZIlly1gegkgqy1tbj314t0u013u.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-04-microkernel-plugin/007S8ZIlly1gegkgqy1tbj314t0u013u.jpg)


registerExtension 负责注册插件。invokeExtension 负责按照名称调用插件，并且传入 Tcb 对象。


根据[tcb 扩展能力文档](https://docs.cloudbase.net/extension/abilities/image-examination.html)，可以看到，插件的注册是在`init`之后：


```typescript
const extCi = require("@cloudbase/extension-ci");
tcb.init({
    env: "您的环境ID"
});
tcb.registerExtension(extCi);
```


对于扩展@cloudbase/extension-ci 来说，它需要暴露 name 属性，以及 invoke 方法。从 npm 下载的源码中可以看到：


```typescript
function invoke(opts, tcb) {
    // ... 具体的业务逻辑
}
exports.name = "CloudInfinite";
exports.invoke = invoke;
```


name 属性作为拓展的标识，在核心系统调用时需要用到；invoke 函数封装具体的业务逻辑，tcb 参数由核心系统注入，可以调用其上方法，例如云函数、云数据库读写、云存储等等。


看到这里，应该可以体会到微内核+插件化的设计的优点了。在 js 中，配合 es6 module，还能起到“源码瘦身、按需加载”的作用。


