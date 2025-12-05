---
title: "从Web开发的角度聊聊MVC、MVP和MVVM"
date: 2020-02-11
permalink: /2020-02-11-mvc-mvp-mvvm/
---
本文从 Web 开发的角度聊聊 MVC、MVP 和 MVVM 这三种架构模式。


## 什么是 M、V？


M、V 是这三种架构模式中的共同含有的部分，M 是 Model 的缩写，代表“数据模型”；V 是 View 的缩写，代表“视图”。


这三种架构设计中，都对 M 和 V 进行了分离，Model 掌握数据源，View 负责视图展示。而剩下的部分（MVC 中的 C、MVP 中的 P、MVVM 中的 VM），就是不同架构中对 M 与 V 之间“交互”的特色处理。


## MVC


MVC 中的 C 是 Controler 的缩写，代表“控制器”，它的职责是消息处理。这里的“消息”在不同情况下，有不同的语义。在前端，消息指的是用户对于视图的操作；在后端，消息指的是来自客户端的 rest api 请求。


对于 View 来说，它不是和 Model 完全分离的。如果用户的操作是访问数据，那么可以在 View 中向 Model 要数据；如果用户的操作是更新数据，那么需要统一交给 可以看出，MVC 的不足是 View 和 Controler 来处理，并且 可以看出，MVC 的不足是 View 和 Controler 在处理完成后，会有机制通知 View，一般采用“观察监听”设计模式。


三者之间的关系如下图所示：


![0082zybply1gbspuq9zvwj30h40awq3s.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-02-11-mvc-mvp-mvvm/0082zybply1gbspuq9zvwj30h40awq3s.jpg)


如果想看 Nodejs 的 demo，请参考[《使用 Node.js 实现简易 MVC 框架》](https://www.cnblogs.com/SheilaSun/p/7294706.html)这篇文章。


## MVP


可以看出，MVC 的不足是 View 和 Model 之间不是严格意义的完全分离。MVP 正是对 MVC 这一点做出了改进。


MVP 中的 P 是 Presenter 的缩写，代表“展示器”。所有的消息（客户端请求、用户事件）都统一交给 Presenter 来处理，由 Presenter 来向 Model 进行数据查询或者更新。而 Presenter 和 View 之间，一般会约定好接口调用的格式。


三者之间的关系如下图所示：


![0082zybply1gbspv73nyfj30h80b0753.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-02-11-mvc-mvp-mvvm/0082zybply1gbspv73nyfj30h80b0753.jpg)


## MVVM


可以看出，MVP 虽然实现了 V 和 M 的分离，但是开发者必须提前规定 P 和 V 的交互接口，对开发来说并不友好。有没有办法能够实现，当 Model 发生改变的时候，立即就下发到视图，并且实现视图更新呢？


MVVM 通过“双向绑定”实现了这个要求。MVVM 中的 VM 是 View Model 的缩写，代表“数据模型”。


前端框架 Vuejs 就使用了这种设计，使得开发者用起来非常方便。开发者只需要关注 View Model 和 Model 即可，不再需要对 View 进行显式手动操作：用户事件导致的 View 变动会自然反映在 ViewModel 上，ViewModel 中的数据操作也会自动反映在 View 上。


它们的关系如下图所示：


![0082zybply1gbsqsa790fj30ik05iwf1.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-02-11-mvc-mvp-mvvm/0082zybply1gbsqsa790fj30ik05iwf1.jpg)


## Vue.js 和 React.js 都是MVVM吗？


结论：React.js 不是，Vue.js 是。MVVM一个最显著的特征：双向绑定。


在 Vue.js 中，更新 ViewModel 上的数据，可以反应到 View 上；View上绑定了 ViewModel，比如 `input` 绑定了属性 [`data.name`](http://data.name/) ，那么操作 View 时，结果也能反应到 ViewModel 上。


在 React.js 没有这个，React.js 做的事情简单来说就下面的公式：


```javascript
view = render(model)
```


同样的例子，在 React.js 中，如果 `input` 属性变了，那么需要绑定 `onChange` 回调函数，在回调函数中手动调用 `setState` 来更新 Model。**和Vue.js比起来，并不是双向绑定，需要手动实现从 view ⇒ model 。**


