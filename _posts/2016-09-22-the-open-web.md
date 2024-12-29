---
layout: post
title: "Web 在继续离我们远去"
subtitle: "After the release of Wechat Mini-Program"
author: "Hux"
header-img: "img/post-bg-web.jpg"
header-mask: 0.4
tags:
  - Web
  - 微信
---

> 本文首发于我的知乎专栏 [The Little Programmer](https://zhuanlan.zhihu.com/p/22561084)，转载请保留链接 ;)

今天微信又刷爆了我的朋友圈 —— 小程序，之前传说的应用号。

不过这篇不谈小程序的技术细节，也不去猜测（因为知道得很清楚……），

也不谈小程序会对中国互联网带来什么影响（自有产品经理会来谈……），

我们说说 Web，the Web。

我们常说的 Web，其实是 World Wide Web 的简称 the Web 的简称。

跟 H5 一样，这货是个简称的简称，所以简到最后就没人知道它本身是个什么意思了。

不要说中国老百姓分不清万维网和互联网了，美国老百姓也一样分不清 Web 和 Internet，

很多不求甚解的从业人士也好不到哪去，Web 常年在技术文章中被翻译得不知所云。

中文世界里把这件事讲得最清楚也最在乎的，非 [@不鳥萬如一](//www.zhihu.com/people/6bec872206d9884cd9535841b6a1f510) 莫属了。

比如在[《一天世界》博客：微信并不是在「管理」外部链接，因为微信公众号在事实上（de facto）不允许任何外部链接 - 不鳥萬通讯 - 知乎专栏](https://zhuanlan.zhihu.com/p/20747514) 里他写到：

> 中文世界一直混淆[互联网](https://en.wikipedia.org/wiki/Internet)（internet）和[万维网](https://en.wikipedia.org/wiki/World_Wide_Web)（web）。人们念兹在兹的「互联网开放精神」，实乃万维网的开放精神。万维网的开放主要就体现在一点：**任何万维网上的文章之间都可以通过网址随意互相链接**。如果我想在文章里介绍 UbuWeb 这个网站，我就可以直接在 [UbuWeb](https://ubu.com/) 这六个字母上添加它的网址 ubu.com。妳或许觉得这是废话，但在微信公众号的文章里妳做不到；妳只能添加微信生态圈内的链接，比如这个：[https://weixin.qq.com/cgi-bin/readtemplate?t=weixin_external_links_content_management_specification](https://weixin.qq.com/cgi-bin/readtemplate%3Ft%3Dweixin_external_links_content_management_specification)（即上述《规范》的链接）

所以如一卸了微信（ [告别微信 一天世界](https://blog.yitianshijie.net/2016/02/21/byebye-wechat/) ）还写了：[微信——事实上的局域网](https://blog.yitianshijie.net/2015/11/16/wechat-de-facto-lan/) ，嗯，作为一个愈发对 Open Web 这件事 hardcore 的人来说，我是认同的。

如一最在乎的可能是文章，而我更在乎的是应用，Web App。

所谓 Web App，是 Web 的一种进化：从提供文本信息（超文本）到多媒体（超媒体）到提供软件应用服务。硬核的翻译过来大概是“基于万维网的应用”，比如你在 Web 浏览器中使用的 Youtube、Twitter、Medium、Github 等等，**它们之间仍然可以通过网址（URL）随意互相链接，遵循 Web 开放标准，并且你几乎可以在任何一个具备浏览器的平台上使用这项服务，因此 Web App 同样是开放的。**

如果你听说过 Google 的 Progressive Web Apps，它其实代表的是 Progressive Open Web Apps，只是这样实在太长太啰嗦了。

毕竟，Web 的概念里理应包含着 Open。

（这篇文章的本意并不是为了 advocate PWA，但如果你对 PWA 有兴趣，欢迎阅读： [黄玄：下一代 Web 应用模型 — Progressive Web App​zhuanlan.zhihu.com!](https://zhuanlan.zhihu.com/p/25167289)

如果说 Hybrid 架构还只是 Web 理想主义的一次让步，那么 React Native 的出现则无疑让部分人的信仰崩塌，然后是 Weex，然后可能是你们猜的微信。

眼见 “以 Web 范式为 Native 平台进行开发” 的方式越来越火，虽然受益的好像是 Web 前端从业人员，可我却不知该不该开心。

我不是说它们是“错误的技术方向”，从实用主义来说它们很棒，很解决问题。

**但是，无论他们长得有多像 Web，他们都不是 Open Web 平台的一员。**

RN/Weex 根本没有 URL（别跟我说 Universal Links 或 App Links，URL 和 URI 是不同的）

而微信从 JS-SDK 开始，便已经是一个封闭生态了。

这种势头虽然缘起于 Facebook，却更可能在中国撒起野来。

英文世界里对这类事情敏感/hardcore 的人很多，比如写了 [Regressive Web Apps](https://adactio.com/journal/10708) 的 Jeremy Keith，因为 PWA 对 URL 不够友好的事情跟 Chrome 开发老大 Alex 吵了一架，而 Alex 也急得说出了：

> so, your choices are to think that I have a secret plan to kill URLs, or conclude I’m still Team Web.

要知道，Alex 带着 Chrome 搞 PWA 的原因就是看不爽 Hybrid 破坏了 Open Web。

倘若 Twitter/FB 跟微信一样连链接还不让随便链，大概都得弃用 Twitter，然后像如一一样火冒三丈的写一篇 Byebye Twitter/FB。

而国内天天鼓吹得什么 XX 助力 HTML5 生态，却不知大部分时候这些所谓 “HTML5 生态” 都是和 Web 生态背道而驰的，高下立判。

我开始有些语无伦次了。

在这个 HTML5 与 Web 被极度误用的中文世界里，我也不知道该如何呐喊了。

我只知道，当 Web 只能作为 Native 的 "Markup Language" 存活时，Web 也就不复存在了。

当大家都不跟随 Web 标准自造一套时，Web 所谓的跨平台特性也就烟消云散了。

我之前写过的，Chrome 产品 Leader Rahul 也在 I/O 上说过得：

Web 的 Dicoverable、Linkable、Low Friction、Broad Reach 等等，这些都不是 Web 的本质，**Web 的本质是 Open（开放）与 Decentralized （去中心化），这才是万维网（WWW）的初衷，这才是所有这些特性能成立的前提。**

Open Web 的信仰让浏览器厂商们重新走到了一起，他们在问你:

**Hey, can we make the web great again?**
