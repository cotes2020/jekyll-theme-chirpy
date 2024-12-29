---
layout:     post
title:      "下一代 Web 应用模型 —— Progressive Web App"
subtitle:   "The Next Generation Application Model For The Web - Progressive Web App"
date:       2017-02-09 12:00:00
author:     "Hux"
header-img: "img/post-bg-nextgen-web-pwa.jpg"
header-mask: 0.3
catalog:    true
tags:
    - Web
    - PWA
---


> 今年 9 月份的时候，《程序员》杂志社就邀请我写一篇关于 PWA 的文章。后来花式拖稿，拖过了 10 月的 QCon，11 月的 GDG DevFest，终于在 12 月把这篇长文熬了出来。几次分享的不成熟，这次的结构算是比较满意了。「 可能是目前中文世界里对 PWA 最全面详细的长文了」，希望你能喜欢。<br><br>
> 本文首发于 [CSDN](http://geek.csdn.net/news/detail/135595) 与《程序员》2017 年 2 月刊，同步发布于 [Hux Blog](https://huangxuan.me)、[前端外刊评论 - 知乎专栏](https://zhuanlan.zhihu.com/FrontendMagazine)，转载请保留链接 ;)


## 下一代 Web 应用？

近年来，Web 应用在整个软件与互联网行业承载的责任越来越重，软件复杂度和维护成本越来越高，Web 技术，尤其是 Web 客户端技术，迎来了爆发式的发展。

包括但不限于基于 Node.js 的前端工程化方案；诸如 Webpack、Rollup 这样的打包工具；Babel、PostCSS 这样的转译工具；TypeScript、Elm 这样转译至 JavaScript 的编程语言；React、Angular、Vue 这样面向现代 web 应用需求的前端框架及其生态，也涌现出了像[同构 JavaScript][1]与[通用 JavaScript 应用][2]这样将服务器端渲染（Server-side Rendering）与单页面应用模型（Single-page App）结合的 web 应用架构方式，可以说是百花齐放。

但是，Web 应用在移动时代并没有达到其在桌面设备上流行的程度。究其原因，尽管上述的各种方案已经充分利用了现有的 JavaScript 计算能力、CSS 布局能力、HTTP 缓存与浏览器 API 对当代基于 [Ajax][3] 与[响应式设计][4]的 web 应用模型的性能与体验带来了工程角度的巨大突破，我们仍然无法在不借助原生程序辅助浏览器的前提下突破 web 平台本身对 web 应用固有的桎梏：**客户端软件（即网页）需要下载所带来的网络延迟；与 Web 应用依赖浏览器作为入口所带来的体验问题。**

![](/img/in-post/post-nextgen-web-pwa/PWAR-007.jpeg)
*Web 与原生应用在移动平台上的使用时长对比 [图片来源: Google][i2]*

在桌面设备上，由于网络条件稳定，屏幕尺寸充分，交互方式趋向于多任务，这两点造成的负面影响对比 web 应用免于安装、随叫随到、无需更新等优点，瑕不掩瑜。但是在移动时代，脆弱的网络连接与全新的人机交互方式使得这两个问题被无限放大，严重制约了 web 应用在移动平台的发展。在用户眼里，原生应用不会出现「白屏」，清一色都摆在主屏幕上；而 web 应用则是浏览器这个应用中的应用，使用起来并不方便，而且加载也比原生应用要慢。

Progressive Web Apps（以下简称 PWA）以及构成 PWA 的一系列关键技术的出现，终于让我们看到了彻底解决这两个平台级别问题的曙光：能够显著提高应用加载速度、甚至让 web 应用可以在离线环境使用的 Service Worker 与 Cache Storage；用于描述 web 应用元数据（metadata）、让 web 应用能够像原生应用一样被添加到主屏、全屏执行的 Web App Manifest；以及进一步提高 web 应用与操作系统集成能力，让 web 应用能在未被激活时发起推送通知的 Push API 与 Notification API 等等。

将这些技术组合在一起会是怎样的效果呢？「印度阿里巴巴」 —— [Flipkart][17] 在 2015 年一度关闭了自己的移动端网站，却在年底发布了现在最为人津津乐道的 PWA 案例 *FlipKart Lite*，成为世界上第一个支撑大规模业务的 PWA。发布的一周后它就亮相于 [Chrome Dev Summit 2015][15] 上，笔者当时就被惊艳到了。为了方便各媒介上的读者观看，笔者做了几幅图方便给大家介绍：

![](/img/in-post/post-nextgen-web-pwa/flipkart-1.jpeg)
*图片来源: Hux & [Medium.com][i3]*

当浏览器发现用户[需要][16] Flipkart Lite 时，它就会提示用户「嘿，你可以把它添加至主屏哦」（用户也可以手动添加）。这样，Flipkart Lite 就会像原生应用一样在主屏上留下一个自定义的 icon 作为入口；与一般的书签不同，当用户点击 icon 时，Flipkat Lite 将直接全屏打开，不再受困于浏览器的 UI 中，而且有自己的启动屏效果。


![](/img/in-post/post-nextgen-web-pwa/flipkart-2.jpeg)
*图片来源: Hux & [Medium.com][i3]*

更强大的是，在无法访问网络时，Flipkart Lite 可以像原生应用一样照常执行，还会很骚气的变成黑白色；不但如此，曾经访问过的商品都会被缓存下来得以在离线时继续访问。在商品降价、促销等时刻，Flipkart Lite 会像原生应用一样发起推送通知，吸引用户回到应用。

**无需担心网络延迟；有着独立入口与独立的保活机制。**之前两个问题的一并解决，宣告着 web 应用在移动设备上的浴火重生：满足 PWA 模型的 web 应用，将逐渐成为移动操作系统的一等公民，并将向原生应用发起挑战与「复仇」。

更令笔者兴奋的是，就在今年 11 月的 [Chrome Dev Summit 2016][18] 上，Chrome 的工程 VP Darin Fisher 介绍了 Chrome 团队正在做的一些实验：把「添加至主屏」重命名为「安装」，被安装的 PWA 不再仅以 widget 的形式显示在桌面上，而是真正做到与所有原生应用平级，一样被收纳进应用抽屉（App Drawer）里，一样出现在系统设置中 🎉🎉🎉。

![](/img/in-post/post-nextgen-web-pwa/flipkart-3.jpeg)
*图片来源: Hux & [@adityapunjani][i4]*

图中从左到右分别为：类似原生应用的安装界面；被收纳在应用抽屉里的 Flipkart Lite 与 Hux Blog；设置界面中并列出现的 Flipkart 原生应用与 Flipkart Lite PWA （可以看到 PWA 巨大的体积优势）

**笔者相信，PWA 模型将继约 20 年前横空出世的 Ajax 与约 10 年前风靡移动互联网的响应式设计之后，掀起 web 应用模型的第三次根本性革命，将 web 应用带进一个全新的时代。**

## PWA 关键技术的前世今生

### [Web App Manifest][spec1]

Web App Manifest，即通过一个清单文件向浏览器暴露 web 应用的元数据，包括名字、icon 的 URL 等，以备浏览器使用，比如在添加至主屏或推送通知时暴露给操作系统，从而增强 web 应用与操作系统的集成能力。

让 web 应用在移动设备上的体验更接近原生应用的尝试其实早在 2008 年的 [iOS 1.1.3 与 iOS 2.1.0 ][q37]时就开始了，它们分别为 web 应用增加了对自定义 icon 和全屏打开的支持。

![](/img/in-post/post-nextgen-web-pwa/ios2-a2hs.gif)
*图片来源: [appleinsider.com][i1]*

但是很快，随着越来越多的私有平台通过 `<meta>`/`<link>` 标签来为 web 应用添加「私货」，`<head>` 很快就被塞满了：

```html
<!-- Add to homescreen for Safari on iOS -->
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black">
<meta name="apple-mobile-web-app-title" content="Lighten">

<!-- Add to homescreen for Chrome on Android -->
<meta name="mobile-web-app-capable" content="yes">
<mate name="theme-color" content="#000000">

<!-- Icons for iOS and Android Chrome M31~M38 -->
<link rel="apple-touch-icon-precomposed" sizes="144x144" href="images/touch/apple-touch-icon-144x144-precomposed.png">
<link rel="apple-touch-icon-precomposed" sizes="114x114" href="images/touch/apple-touch-icon-114x114-precomposed.png">
<link rel="apple-touch-icon-precomposed" sizes="72x72" href="images/touch/apple-touch-icon-72x72-precomposed.png">
<link rel="apple-touch-icon-precomposed" href="images/touch/apple-touch-icon-57x57-precomposed.png">

<!-- Icon for Android Chrome, recommended -->
<link rel="shortcut icon" sizes="196x196" href="images/touch/touch-icon-196x196.png">

<!-- Tile icon for Win8 (144x144 + tile color) -->
<meta name="msapplication-TileImage" content="images/touch/ms-touch-icon-144x144-precomposed.png">
<meta name="msapplication-TileColor" content="#3372DF">

<!-- Generic Icon -->
<link rel="shortcut icon" href="images/touch/touch-icon-57x57.png">
```

显然，这种做法并不优雅：分散又重复的元数据定义多余且难以维持同步，与 html 耦合在一起也加重了浏览器检查元数据未来变动的成本。与此同时，社区里开始出现使用 manifest 文件以中心化地描述元数据的方案，比如 [Chrome Extension、 Chrome Hosted Web Apps (2010)][12] 与 [Firefox OS App Manifest (2011)][13] 使用 JSON；[Cordova][19] 与 [Windows Pinned Site][20] 使用 XML；

2013 年，W3C WebApps 工作组开始对基于 JSON 的 Manifest 进行标准化，于同年年底发布[第一份公开 Working Draft][14]，并逐渐演化成为今天的 W3C Web App Manifest：

```json
{
  "short_name": "Manifest Sample",
  "name": "Web Application Manifest Sample",
  "icons": [{
      "src": "launcher-icon-2x.png",
      "sizes": "96x96",
      "type": "image/png"
   }],
  "scope": "/sample/",
  "start_url": "/sample/index.html",
  "display": "standalone",
  "orientation": "landscape"
  "theme_color": "#000",
  "background_color": "#fff",
}
```
```html
<!-- document -->
<link rel="manifest" href="/manifest.json">
```

诸如 `name`、`icons`、`display` 都是我们比较熟悉的，而大部分新增的成员则为 web 应用带来了一系列以前 web 应用想做却做不到（或在之前只能靠 hack）的新特性：

- `scope`：定义了 web 应用的浏览作用域，比如作用域外的 URL 就会打开浏览器而不会在当前 PWA 里继续浏览。
- `start_url`：定义了一个 PWA 的入口页面。比如说你添加 [Hux Blog][21] 的任何一个文章到主屏，从主屏打开时都会访问 [Hux Blog][21] 的主页。
- `orientation`：终于，我们可以锁定屏幕旋转了（喜极而泣…）
- `theme_color`/`background_color`：主题色与背景色，用于配置一些可定制的操作系统 UI 以提高用户体验，比如 Android 的状态栏、任务栏等。

这个清单的成员还有很多，比如用于声明「对应原生应用」的 `related_applications` 等等，本文就不一一列举了。作为 PWA 的「户口本」，承载着 web 应用与操作系统集成能力的重任，Web App Manifest 还将在日后不断扩展，以满足 web 应用高速演化的需要。



### [Service Worker][spec2]

我们原有的整个 Web 应用模型，都是构建在「用户能上网」的前提之下的，所以一离线就只能玩小恐龙了。其实，对于「让 web 应用离线执行」这件事，Service Worker 至少是 web 社区的第三次尝试了。

故事可以追溯到 2007 年的 [Google Gears][48]：为了让自家的 Gmail、Youtube、Google Reader 等 web 应用可以在本地存储数据与离线执行，Google 开发了一个浏览器拓展来增强 web 应用。Google Gears 支持 IE 6、Safari 3、Firefox 1.5 等浏览器；要知道，那一年 Chrome 都还没出生呢。

在 Gears API 中，我们通过向 LocalServer 模块提交一个缓存文件清单来实现离线支持：

```javascript
// Somewhere in your javascript
var localServer = google.gears.factory.create("bata.localserver");
var store = localServer.createManagedStore(STORE_NAME);
store.manifestUrl = "manifest.json"
```
```js
// manifest.json
{
  "betaManifestVersion": 1,
  "version": "1.0",
  "entries": [
    { "url": "index.html" }, 
    { "url": "main.js" }
  ]
}
```

是不是感到很熟悉？好像 [HTML5 规范][spec11]中的 Application Cache 也是类似的东西？

```html
<html manifest="cache.appcache">
```
```
CACHE MANIFEST

CACHE:
index.html
main.js
```

是的，Gears 的 LocalServer 就是后来大家所熟知的 App Cache 的前身，大约从 [2008][spec10] 年开始 W3C 就开始尝试将 Gears 进行标准化了；除了 LocalServer，Gears 中用于提供并行计算能力的 WorkerPool 模块与用于提供本地数据库与 SQL 支持的 Database 模块也分别是日后 Web Worker 与 Web SQL Database（后被废弃）的前身。

HTML5 App Cache 作为第二波「让 web 应用离线执行」的尝试，确实也服务了比如 Google Doc、尤雨溪早年作品 HTML5 Clear、以及一直用 web 应用作为自己 iOS 应用的 FT.com（Financial Times）等不少 web 应用。那么，还有 Service Worker 什么事呢？  

是啊，如果 App Cache 没有被设计得[烂到完全不可编程、无法清理缓存、几乎没有路由机制、出了 Bug 一点救都没有][s12]，可能就真没 Service Worker 什么事了。[App Cache 已经在前不久定稿的 HTML5.1 中被拿掉了，W3C 为了挽救 web 世界真是不惜把自己的脸都打肿了……][s13]

时至今日，我们终于迎来了 Service Worker 的曙光。简单来说，Service Worker 是一个可编程的 Web Worker，它就像一个位于浏览器与网络之间的客户端代理，可以拦截、处理、响应流经的 HTTP 请求；配合随之引入 Cache Storage API，你可以自由管理 HTTP 请求文件粒度的缓存，这使得 Service Worker 可以从缓存中向 web 应用提供资源，即使是在离线的环境下。


![](/img/in-post/post-nextgen-web-pwa/sw-sw.png)
*Service Worker 就像一个运行在客户端的代理*

比如说，我们可以给网页 `foo.html` 注册这么一个 Service Worker，它将劫持由 `foo.html` 发起的一切 HTTP 请求，并统统返回未设置 `Content-Type` 的 `Hello World!`：

```javascript
// sw.js
self.onfetch = (e) => {
  e.respondWith(new Response('Hello World!'))
}
```

Service Worker 第一次发布于 2014 年的 Google IO 上，目前已处于 W3C 工作草案的状态。其设计吸取了 Application Cache 的失败经验，作为 web 应用的开发者的你有着完全的控制能力；同时，它还借鉴了 Chrome 多年来在 Chrome Extension 上的设计经验（Chrome Background Pages 与 Chrome Event Pages），采用了基于「事件驱动」的唤醒机制，以大幅节省后台计算的能耗。比如上面的 `fetch` 其实就是会唤醒 Service Worker 的事件之一。

![](/img/in-post/post-nextgen-web-pwa/sw-lifecycle.png)
*Service Worker 的生命周期*

除了类似 `fetch` 这样的功能事件外，Service Worker 还提供了一组生命周期事件，包括安装、激活等等。比如，在 Service Worker 的「安装」事件中，我们可以把 web 应用所需要的资源统统预先下载并缓存到 Cache Storage 中去：

```javascript
// sw.js
self.oninstall = (e) => {
  e.waitUntil(
    caches.open('installation')
      .then(cache =>  cache.addAll([
        './',
        './styles.css',
        './script.js'
      ]))
  )
});
```

这样，当用户离线，网络无法访问时，我们就可以从缓存中启动我们的 web 应用：

```javascript
//sw.js
self.onfetch = (e) => {
  const fetched = fetch(e.request)
  const cached = caches.match(e.request)

  e.respondWith(
    fetched.catch(_ => cached)
  )
}
```

可以看出，Service Worker 被设计为一个相对底层（low-level）、高度可编程、子概念众多，也因此异常灵活且强大的 API，故本文只能展示它的冰山一角。出于安全考虑，注册 Service Worker 要求你的 web 应用部署于 HTTPS 协议下，以免利用 Service Worker 的中间人攻击。笔者在今年 GDG 北京的 DevFest 上分享了 [Service Worker 101][b0]，涵盖了 Service Worker 譬如「网络优先」、「缓存优先」、「网络与缓存比赛」这些更复杂的缓存策略、学习资料、以及[示例代码][29]，可以供大家参考。


![](/img/in-post/post-nextgen-web-pwa/sw-race.png)
*Service Worker 的一种缓存策略：让网络请求与读取缓存比赛*

你也可以尝试在支持 PWA 的浏览器中访问笔者的博客 [Hux Blog][21]，感受 Service Worker 的实际效果：所有访问过的页面都会被缓存并允许在离线环境下继续访问，所有未访问过的页面则会在离线环境下展示一个自定义的离线页面。

在笔者看来，**Service Worker 对 PWA 的重要性相当于 `XMLHTTPRequest` 之于 Ajax，媒体查询（Media Query）之于响应式设计，是支撑 PWA 作为「下一代 web 应用模型」的最核心技术。**由于 Service Worker 可以与包括 Indexed DB、Streams 在内的大部分 DOM 无关 API 进行交互，它的潜力简直无可限量。笔者几乎可以断言，Service Worker 将在未来十年里成为 web 客户端技术工程化的兵家必争之地，带来「离线优先（Offline-first）」的架构革命。



### Push Notification

PWA 推送通知中的「推送」与「通知」，其实使用的是两个不同但又相得益彰的 API：

[Notification API][spec4] 相信大家并不陌生，它负责所有与通知本身相关的机制，比如通知的权限管理、向操作系统发起通知、通知的类型与音效，以及提供通知被点击或关闭时的回调等等，目前国内外的各大网站（尤其在桌面端）都有一定的使用。Notification API 最早应该是在 [2010][22] 年前后由 Chromium 提出[草案][spec7]以 `webkitNotifications` 前缀方式实现；随着 2011 年进入标准化；2012 年在 Safari 6（Mac OSX 10.8+）上获得支持；2015 年 Notification API 成为 [W3C Recommendation][spec8]；2016 年 [Edge 的支持][23]；Web Notifications 已经在桌面浏览器中获得了全面支持（Chrome、Edge、Firefox、Opera、Safari）的成就。

[Push API][spec3] 的出现则让推送服务具备了向 web 应用推送消息的能力，它定义了 web 应用如何向推送服务发起订阅、如何响应推送消息，以及 web 应用、应用服务器与推送服务之间的鉴权与加密机制；由于 Push API 并不依赖 web 应用与浏览器 UI 存活，所以即使是在 web 应用与浏览器未被用户打开的时候，也可以通过后台进程接受推送消息并调用 Notification API 向用户发出通知。值得一提的是，Mac OSX 10.9 Mavericks 与 Safari 7 在 2013 年就发布了自己的私有推送支持，基于 APNS 的 [Safari Push Notifications][24]。

在 PWA 中，我们利用 Service Worker 的后台计算能力结合 Push API 对推送事件进行响应，并通过 Notification API 实现通知的发出与处理：

```javascript
// sw.js
self.addEventListener('push', event => {
  event.waitUntil(
    // Process the event and display a notification.
    self.registration.showNotification("Hey!")
  );
});

self.addEventListener('notificationclick', event => {  
  // Do something with the event  
  event.notification.close();  
});

self.addEventListener('notificationclose', event => {  
  // Do something with the event  
});
```

对于 Push Notification，笔者的几次分享中一直都提的稍微少一些，一是因为 Push API 还处于 Editor Draft 的状态，二是目前浏览器与推送服务间的协议支持还不够成熟：Chrome（与其它基于 Blink 的浏览器）在 Chromium 52 之前只支持基于 Google 私有的 GCM/FCM 服务进行通知推送。不过好消息是，继 Firefox 44 之后，Chrome 52 与 Opera 39 也紧追其后实现了正在由 IETF 进行标准化的 [Web 推送协议（Web Push Protocol）][spec5]。


如果你已经在使用 Google 的云服务（比如 Firebase），并且主要面向的是海外用户，那么在 web 应用上支持基于 GCM/FCM 的推送通知并不是一件费力的事情，笔者推荐你阅读一下 Google Developers 的[系列文章][25]，很多国外公司已经玩起来了。



## 从 Hybrid 到 PWA，从封闭到开放

2008 年，当移动时代来临，[唱衰移动 Web 的声音][q17]开始出现，而浏览器的进化并不能跟上时，来自 Nitobi 的 Brian Leroux 等人创造了 [Phonegap][10]，希望它能以 Polyfill 的形式、弥补目前浏览器与移动设备间的「鸿沟」，从此开启了[混合应用（Hybrid Apps）][26]的时代。

几年间，[Adobe AIR][5]、[Windows Runtime Apps][6]、[Chrome Apps][7]、[Firefox OS][8]、[WebOS][9]、[Cordova/Phonegap][10]、[Electron][11] 以及国内比如微信、淘宝，无数的 Hybrid 方案拔地而起，让 web 开发者可以在继续使用 web 客户端技术的同时，做到一些只有原生应用才能做到的事情，包括访问一些设备与操作系统 API，给用户带来更加 「Appy」 的体验，以及进入 App Store 等等。

![](/img/in-post/post-nextgen-web-pwa/qcon-hybridzation.png)
*众多的 Hybrid 方案*

PWA 作为一个涵盖性术语，与过往的这些或多或少通过私有平台 API 增强 web 应用的尝试最大的不同，在于构成 PWA 的每一项基本技术，都已经或正在被 IETF、ECMA、W3C 或 WHATWG 标准化，不出意外的话，它们都将被纳入开放 web 标准，并在不远的将来得到所有浏览器与全平台的支持。我们终于可以逃出 App Store 封闭的秘密花园，重新回到属于 web 的那片开放自由的大地。

有趣的是，从上文中你也可以发现，组成 PWA 的各项技术的草案正是由上述各种私有方案背后的浏览器厂商或开发者直接贡献或间接影响的。可以说，PWA 的背后并不是某一家或两家公司，而是整个 web 社区与整个 web 规范。**正是因为这种开放与去中心化的力量，使得万维网（World Wide Web）能够成为当今世界上跨平台能力最强、且几乎是唯一一个具备这种跨平台能力的应用平台。**

[「我们相信 Web，是因为相信它是解决设备差异化的终极方案；我们相信，当 Web 在今天做不到一件事的时候，是因为它还没来得及去实现，而不是因为他做不到。而 Phonegap，它的终极目的就是消失在 Web 标准的背后。」][27]

在不丢失 web 的开放灵魂，在不需要依靠 Hybrid 把应用放在 App Store 的前提下，让 web 应用能够渐进式地跳脱出浏览器的标签，变成用户眼中的 App。这是 Alex Russell 在 2015 年提出 PWA 概念的[原委][28]。

而又正因为 web 是一个整体，PWA 可以利用的技术远不止上述的几个而已：Ajax、响应式设计、JavaScript 框架、ECMAScript Next、CSS Next、Houdini、Indexed DB、Device APIs、Web Bluetooth、Web Socket、Web Payment、[孵化][spec6]中的 [Background Sync API][30]、[Streams][spec9]、WebVR……开放 Web 世界 27 年来的发展以及未来的一切，都与 PWA 天作之合。


## 鱼与熊掌的兼得

经过几年来的摸索，整个互联网行业仿佛在「Web 应用 vs. 原生应用」这个问题上达成了共识：

- web 应用是鱼：迭代快，获取用户成本低；跨平台强体验弱，开发成本低。**适合拉新**。
- 原生应用是熊掌：迭代慢，获取用户成本高；跨平台弱体验强，开发成本高。**适合保活**。

要知道，虽然用户花在原生应用上的时间要明显多于 web 应用，但其中[有 80% 的时间是花在前五个应用中的][31]。[调查显示，美国有一半的智能手机用户平均每月新 App 安装量为零][32]，而月均网站访问量却有 100 个，更别提 Google Play 上[有 60% 的应用从未被人下载过了][33]。于是，整个行业的产品策略清一色地**「拿鱼换熊掌」**，比如笔者的老东家阿里旅行（飞猪旅行），web 应用布满阿里系各种渠道，提供「优秀的第一手体验」，等你用的开心了，再引诱你去下载安装原生应用。

![](/img/in-post/post-nextgen-web-pwa/PWAR-014+PWA.jpeg)
*原生应用、当代 Web 与 PWA 图片来源: Hux & [Google][i2]*

但是，PWA 的出现，让鱼与熊掌兼得变成了可能 —— 它同时具备了 web 应用与原生应用的优点，有着自己独有的先进性：「浏览器 -> 添加至主屏/安装 -> 具备原生应用体验的 PWA -> 推送通知 -> 具备原生应用体验的 PWA」，PWA 自身就包含着从拉新到保活的闭环。

除此之外，PWA 还继承了 web 应用的另外两大优点：**无需先付出几十兆的下载安装成本即可开始使用**，以及**不需要经过应用超市审核就可以发布新版本**。所以，PWA 可以称得上是一种「流式应用（Streamable App）」与「常青应用（Evergreen App）」


## 未来到来了吗

在笔者分享 PWA 的经历中，最不愿意回答的两个问题莫过于「PWA 已经被广泛支持了吗？」以及「PWA 与 ABCDEFG 这些技术方案相比有什么优劣？」，但是这确实是两个逃不开的问题。

### PWA 的支持情况？

当我们说到 PWA 是否被支持时，其实我们在说的是 PWA 背后的几个关键技术都得到支持了没有。以浏览器内核来划分的话，Blink（Chrome、Oprea、Samsung Internet 等）与 Gecko（Firefox）都已经实现了 PWA 所需的所有关键技术（👏👏👏），并已经开始探寻更多的可能性。EdgeHTML（Edge）[简直积极得不能更积极了][34]，所有的特性都已经处于「正在开发中」的[状态][35]。最大的绊脚石仍然来自于 Webkit（Safari），尤其是在 iOS 上，上述的四个 API 都未得到支持，而且由于平台限制，第三方浏览器也无法在 iOS 上支持。（[什么你说 IE？][42]）

不过，也不要气馁，Webkit 不但在它 [2015 年发布的五年计划][36]里提到了 Service Worker，更是已经在最近实现了 Service Worker 所[依赖][41]的 Request、Response 与 Fetch API，还把 Service Worker 与 Web App Manifest 纷纷[列入了「正在考虑」][37]的 API 中；要知道，Webkit 可是把 Web Components 中的 HTML Imports 直接[列到「不考虑」里去了][38]……（其实 Firefox 也是）

更何况，由于 web 社区一直以来所追求的「渐进增强、优雅降级」，一个 PWA 当然可以在 iOS 环境正常执行。[事实上，华盛顿邮报将网站迁移到 PWA 之后发现，不止是 Android，在 iOS 上也获得了 5 倍的活跃度增长][39]，（无论是不是它们之前的网站写得太烂吧），就算 iOS 现在还不支持 PWA 也[不会怎么样][40]，我们更是有理由相信 PWA 会很快在 iOS 上到来。

### PWA vs. Others

贺老（贺师俊）曾说过：「从纯 Web 到纯 Native，之间有许多可能的点」。当考虑移动应用的技术选型时，除了 Web 与原生应用，我们还有各种不同程度的 Hybrid，还有今年爆发的诸多 JS-to-Native 方案。

虽然我在上文中用了「复仇」这样的字眼，不过无论从技术还是商业的角度，我们都没必要把 web 或是 PWA 放到 Native 的对立面去看。它们当然存在竞争关系，但是更多的时候，web-only 与 app-only 的策略都是不完美的，当公司资源足够的时候，我们通常会选择同时开发两者。[当然，无论与不与原生应用对比，PWA 让 web 应用变得体验更好这件事本身是毋庸置疑的。][43]「不谈场景聊技术都是扯淡」，[我们仍然还是需要根据自己产品与团队的情况来决定对应的技术选型与平台策略，只是 PWA 让 web 应用在面对选型考验时更加强势了而已。][44]


![](/img/in-post/post-nextgen-web-pwa/qcon-trend.png)
*众多的技术选型，以及笔者的一种猜测*

笔者不负责任得做一些猜测：虽然[重量级的 Hybrid 架构与基础设施][45]仍是目前不少场景下最优的解决方案；但是随着移动设备本身的硬件性能提升与新技术的成熟与普及，JS-to-Native 与以 PWA 为首的纯 web 应用，将分别从两个方向挤压 Hybrid 的生存空间，消化当前 Hybrid 架构主要解决的问题；前者将逐渐演化为类似 Xarmarin 这样针对跨平台原生应用开发的解决方案；后者将显著降低当前 Hybrid 架构的容器开发与部署成本，将 Hybrid 返璞归真为简单的 webview 调用。

这种猜测当然不是没有依据的瞎猜，比如前者可以参考阿里巴巴集团级别迁移 Weex 的战略与微信小程序的 roadmap；后者则可以参考当前 Cordova 与 Ionic 两大 Hybrid 社区对 PWA 的热烈反响。

### PWA in China

看看 Google 官方宣传较多的 PWA [案例][47]就会发现，FlipKart、Housing.com 来自印度；Lyft、华盛顿邮报来自北美；唯一来自中国的 AliExpress 主要开展的则是海外业务。

由于中国的特殊性，笔者在[第一次][46]聊到 PWA 时难免表现出了一定程度的悲观：

- 国内较重视 iOS，而 iOS 目前还不支持 PWA。
- 国内的 Android 实为「安卓」，不自带 Chrome 是一，可能还会有其他兼容问题。
- 国内厂商可能并不会像三星那样对推动自家浏览器支持 PWA 那么感兴趣。
- 依赖 GCM 推送的通知不可用，Web Push Protocol 还没有国内的推送服务实现。
- 国内 webview 环境较为复杂（比如微信），黑科技比较多。

反观印度，由于 Google 服务健全、标配 Chrome 的 Android 手机市占率非常高，PWA 的用户达到率简直直逼 100%，也难免获得无数好评与支持了。**笔者奢望着本文能对推动 PWA 的国内环境有一定的贡献。**不过无论如何，PWA 在国内的春天可能的确会来得稍微晚一点了。


## 结语

「[我们信仰 Web，不仅仅在于软件、软件平台与单纯的技术][q97]，还在于[『任何人，在任何时间任何地点，都可以在万维网上发布任何信息，并被世界上的任何一个人所访问到。』而这才是 web 的最为革命之处，堪称我们人类，作为一个物种的一次进化。][27]」

请不要让 web 再[继续离我们远去][49]，浏览器厂商们已经重新走到了一起，而下一棒将是交到我们 web 应用开发者的手上。[乔布斯曾相信 web 应用才移动应用的未来][50]，那就让我们用代码证明给这个世界看吧。

**让我们的用户，也像我们这般热爱 web 吧。**

黄玄，于 12 月的北京。

---

*注：在笔者撰文期间，Google 在 Google China Developers Days 上宣布了 developers.google.cn 域名的启用，方便国内开发者访问。对于文中所有链向 developers.google.com 的参考文献，应该都可以在 cn 站点中找到。*


[1]: http://nerds.airbnb.com/isomorphic-javascript-future-web-apps/ "Isomorphic JavaScript: The Future of Web Apps"

[2]: https://medium.com/@mjackson/universal-javascript-4761051b7ae9#.unrzyz3b2 "Universal JavaScript"

[3]: https://en.wikipedia.org/wiki/Ajax_(programming) "Ajax - Wikipedia"

[4]: https://en.wikipedia.org/wiki/Responsive_web_design "Responsive Web Design - Wikipedia"

[5]: http://www.adobe.com/products/air.html "Adobe AIR Application"

[6]: https://msdn.microsoft.com/en-us/library/windows/apps/br211385.aspx "Windows Runtime JS API"

[7]: https://developer.chrome.com/extensions/apps "Chrome Packaged Apps"

[8]: https://developer.mozilla.org/en-US/docs/Archive/Firefox_OS/Firefox_OS_apps/Building_apps_for_Firefox_OS "Firefox OS Packaged Apps"

[9]: http://www.openwebosproject.org/ "Open webOS"

[10]: https://cordova.apache.org/ "Apache Cordova"

[11]: http://electron.atom.io/ "Electron"

[12]: https://developer.chrome.com/extensions/manifest "Chrome Apps Manifest"

[13]: https://developer.mozilla.org/en-US/docs/Archive/Firefox_OS/Firefox_OS_apps/Building_apps_for_Firefox_OS/Manifest "Firefox OS App Manifest"

[14]: https://www.w3.org/TR/2013/WD-appmanifest-20131217/ "Manifest for web apps and bookmarks - First Public Working Draft"

[15]: https://youtu.be/m2a9hlUFRhg "Keynote (Chrome Dev Summit 2015)"

[16]: https://developers.google.com/web/fundamentals/engage-and-retain/app-install-banners/?hl=en "Web App Install Banners - Google Developer"

[17]: https://en.wikipedia.org/wiki/Flipkart "Flipkart - wikipedia"

[18]: https://youtu.be/eI3B6x0fw9s "Keynote (Chrome Dev Summit 2016)"

[19]: http://cordova.apache.org/docs/en/6.x/config_ref/index.html "Config.xml - Apache Cordova"

[20]: https://msdn.microsoft.com/en-us/library/dn320426%28v=vs.85%29.aspx "Browser configuration schema reference - MSDN"

[21]: https://huangxuan.me "Hux Blog"

[22]: https://www.html5rocks.com/en/tutorials/notifications/quick/ "Using the Notification API"

[23]: https://blogs.windows.com/msedgedev/2016/05/16/web-notifications-microsoft-edge/#2VBm890EjvAvUcgE.97

[24]: https://developer.apple.com/notifications/safari-push-notifications/ "Safari Push Notifications"

[25]: https://developers.google.com/web/fundamentals/engage-and-retain/push-notifications/ "Web Push Notifications - Google Developer"

[26]: https://en.wikipedia.org/wiki/Progressive_web_app#Hybrid_Apps

[27]: http://phonegap.com/blog/2012/05/09/phonegap-beliefs-goals-and-philosophy/ "PhoneGap Beliefs, Goals, and Philosophy"

[28]: https://infrequently.org/2015/06/progressive-apps-escaping-tabs-without-losing-our-soul/ "Progressive Web Apps: Escaping Tabs Without Losing Our Soul"

[29]: https://github.com/Huxpro/sw-101-gdgdf

[30]: developers.google.com/web/updates/2015/12/background-sync "Background Sync - Google Developers"

[31]: http://marketingland.com/report-mobile-users-spend-80-percent-time-just-five-apps-116858 "Report: Mobile Users Spend 80 Percent Of Time In Just Five Apps"

[32]: http://www.recode.net/2016/9/16/12933780/average-app-downloads-per-month-comscore "Half of U.S. smartphone users download zero apps per month"

[33]: https://youtu.be/EUthgV-U05w "AdWords for App Promotion - Google"

[34]: https://blogs.windows.com/msedgedev/2016/07/08/the-progress-of-web-apps/ "The Progress of Web Apps - MSEdgeDev Blog"

[35]: https://developer.microsoft.com/en-us/microsoft-edge/platform/status/ "Microsoft Edge web platform features status"

[36]: https://trac.webkit.org/wiki/FiveYearPlanFall2015

[37]: https://webkit.org/status/ "Webkit Feature Status"

[38]: https://webkit.org/status/#specification-web-components "HTML Imports - Not Considering"

[39]: https://cloudfour.com/thinks/why-does-the-washington-posts-progressive-web-app-increase-engagement-on-ios/ "Why does The Washington Post’s Progressive Web App increase engagement on iOS?"

[40]: https://cloudfour.com/thinks/ios-doesnt-support-progressive-web-apps-so-what/ "iOS doesn’t support Progressive Web Apps, so what?"

[41]: https://jakearchibald.github.io/isserviceworkerready/ "Is Service Worker Ready?"

[42]: https://www.microsoft.com/en-us/WindowsForBusiness/End-of-IE-support "Internet Explorer End of Support"

[43]: https://cloudfour.com/thinks/progressive-web-apps-simply-make-sense/?utm_source=mobilewebweekly&utm_medium=email#fn-4857-1 "Progressive Web Apps Simply Make Sense"

[44]: https://medium.com/@owencm/the-surprising-tradeoff-at-the-center-of-question-whether-to-build-an-native-or-web-app-d2ad00c40fb2#.ym83ct2ax "The surprising tradeoff at the center of the question whether to build a Native or Web App"

[45]: http://zhihu.com/question/31316032/answer/75236718

[46]: https://www.zhihu.com/question/46690207/answer/104851767

[47]: https://developers.google.com/web/showcase/ "Case Studies - Google Developers"

[48]: https://en.wikipedia.org/wiki/Google_Gears "Gears - Wikipedia"

[49]: https://zhuanlan.zhihu.com/p/22561084 "Web 在继续离我们远去"

[50]: youtu.be/y1B2c3ZD9fk?t=1h14m48s "WWDC 2017"


[spec1]: https://w3c.github.io/manifest/#use-cases-and-requirements "Web App Manifest"

[spec2]: https://w3c.github.io/ServiceWorker/ "Service Worker"

[spec3]: http://w3c.github.io/push-api/ "Push API"

[spec4]: https://notifications.spec.whatwg.org/ "Notification API"

[spec5]: https://tools.ietf.org/html/draft-ietf-webpush-protocol-12 "Web Push Protocol"

[spec6]: https://wicg.github.io/BackgroundSync/spec/ "Web Background Synchronization - WICG"

[spec7]: http://www.chromium.org/developers/design-documents/desktop-notifications/api-specification "API Specification - The Chromium Projects"

[spec8]: https://www.w3.org/TR/notifications/ "Web Notifications - W3C"

[spec9]: https://streams.spec.whatwg.org/ "Streams"

[spec10]: https://www.w3.org/TR/offline-webapps/ "Offline Web Applications"

[spec11]: https://www.w3.org/TR/2011/WD-html5-20110525/offline.html "HTML5 5.6 Offline Web Applications"


[i1]: http://appleinsider.com/articles/08/10/03/latest_iphone_software_supports_full_screen_web_apps.html

[i2]: https://developers.google.com/web/events/pwaroadshow/

[i3]: https://medium.com/@AdityaPunjani/building-flipkart-lite-a-progressive-web-app-2c211e641883#.hz4d3kw41 "Building Flipkart Lite: A Progressive Web App"

[i4]: https://twitter.com/adityapunjani


[q37]: https://huangxuan.me/pwa-qcon2016/#/37 "PWA@QCon2016 #37"

[q17]: https://huangxuan.me/pwa-qcon2016/#/17 "PWA@QCon2016 #17"

[q97]: https://huangxuan.me/pwa-qcon2016/#/99 "PWA@QCon2016 #97"

[s12]: https://huangxuan.me/sw-101-gdgdf/#/12 "SW-101@DevFest #12"

[s13]: https://huangxuan.me/sw-101-gdgdf/#/13 "SW-101@DevFest #13"

[b0]: https://huangxuan.me/2016/11/20/sw-101-gdgdf/
