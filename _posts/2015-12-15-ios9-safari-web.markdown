---
layout:     post
title:      "「译」iOS 9，为前端世界都带来了些什么？"
subtitle:   "iOS 9, Safari and the Web: 3D Touch, new Responsive Web Design, Native integration and HTML5 APIs"
date:       2015-12-15
author:     "Hux"
header-img: "img/post-bg-ios9-web.jpg"
catalog:    true
tags:
    - Web
    - 译
---

2015 年 9 月，Apple 重磅发布了全新的 iPhone 6s/6s Plus、iPad Pro 与全新的操作系统 watchOS 2 与 tvOS 9（是的，这货居然是第 9 版），加上已经发布的 iOS 9，它们都为前端世界带来了哪些变化呢？作为一个 web 开发者，是时候站在我们的角度来说一说了！


> **注！** 该译文存在大量英文术语，笔者将默认读者知晓 ES6、viewport、native app、webview 等常用前端术语，并不对这些已知术语进行汉语翻译
> 对于新发布或较新的产品名称与技术术语，诸如 Apple Pen、Split View 等专有名词，笔者将在文中使用其英文名，但会尝试对部分名词进行汉语标注
> 另外，出于对 wiki 式阅读的偏爱，笔者为您添加了很多额外的链接，方便您查阅文档或出处


## 简而言之

如果你不想阅读整篇文章，这里为你准备了一个总结：


#### 新的设备特性

* iPhone 6s 与 6s Plus 拥有 **“[3D Touch](http://www.apple.com/iphone-6s/3d-touch/)”**，这是一个全新的硬件特性，它可以侦测压力，是一个可以让你拿到手指压力数据的 API
* iPad Pro 的 viewport 为 1024px，与以往的 iPad 全都不同
* 想在 iPad Pro 上支持新的 Apple Pen？不好意思，目前似乎并没有适用于网站的 API 

#### 新的操作系统特性（与 web 相关的）

* iPad 上的 Safari 现在可以通过 [Split View](https://developer.apple.com/library/prerelease/ios/documentation/WindowsViews/Conceptual/AdoptingMultitaskingOniPad/QuickStartForSlideOverAndSplitView.html#//apple_ref/doc/uid/TP40015145-CH13-SW1)（分屏视图）与其他应用一起使用，这意味着新的 viewport 尺寸将会越来越常见
* 新的 Safari View Controller（[`SFSafariViewController`](https://developer.apple.com/library/prerelease/ios/documentation/SafariServices/Reference/SFSafariViewController_Ref/index.html#//apple_ref/occ/cl/SFSafariViewController)）可以让你在 native app 内提供与 Safari 界面、行为连贯一致的应用内网页浏览体验
* 注意啦！Safari 新加入了 Content Blocker（内容拦截器）。以后，并不是所有的访问都一定会出现在你的 Google Analytics 了
* [Universal Links](https://developer.apple.com/library/prerelease/ios/documentation/General/Conceptual/AppSearch/UniversalLinks.html#//apple_ref/doc/uid/TP40016308-CH12) 可以让应用的拥有者在 iOS 内部“占有”自己的域名。因此，访问 yourdomain.com 将会打开你的应用（类似 Android 的 Intents 机制）
* [App Search（应用搜索）](https://developer.apple.com/library/prerelease/ios/documentation/General/Conceptual/AppSearch/index.html#//apple_ref/doc/uid/TP40016308)：现在，Apple 将会抓取你的网页内容（与 native app 内容）用于 Spotlight 与 Siri 的搜索结果，[想知道你的标签都兼容吗？](https://developer.apple.com/library/prerelease/ios/documentation/General/Conceptual/AppSearch/WebContent.html#//apple_ref/doc/uid/TP40016308-CH8)
* 你的网站现在可以通过 JavaScript API 访问 iCloud 的用户数据

#### 新的 API 支持

* [Performance Timing API](https://developer.mozilla.org/en-US/docs/Web/API/Performance/timing) 在 iOS 9 得到回归
* 关于 HTML5 Video，你现在可以在支持 [Picture in Picture（画中画）](https://developer.apple.com/library/prerelease/ios/documentation/WindowsViews/Conceptual/AdoptingMultitaskingOniPad/QuickStartForPictureInPicture.html#//apple_ref/doc/uid/TP40015145-CH14)的 iPad 设备上提供这项新功能；你的视频甚至可以在 Safari 关闭后继续播放
* 更好的 ES6 支持：classes（类）, computed properties（可计算属性）, template literals（模版字符串）等
* Backdrop CSS filters（背景滤镜）
* CSS @supports 与 CSS Supports JavaScript API 
* CSS Level4 伪选择器
* 用于支持分页内容的 CSS Scroll Snapping
* WKWebView 现在可以访问本地文件了
* 我们仍然需要等待 Push Notification，camera access，Service Workers 这些现代 web API 的到来

#### 新的操作系统

* 新一代 Apple TV 的 **tvOS**： 没有浏览器，也没有 webview。但是 JavaScript、XHR 和 DOM 可以通过一个叫做 TVML 的标记语言来使用
* Apple Watch 的 **watchOS**：完全没有任何浏览器和 webview


> **再注！** 由于原文写于 Apple 发布会之前，为了不让读者感到奇怪，笔者将会对文章进行适当改写与补充，以保证本文的连贯性


## 新的 iOS 设备特性

### iPhones 6s 与 3D Touch

从 web 设计与开发的角度来说，新的 iPhone 6s 与 6s Plus 与之前的版本并没有太多差别。不过，有一个特性注定会吸引我们的目光：**3D Touch**

我们无法确定 Apple 是不是只是重命名了一下 “Force Touch”（用于 Apple Watch、TrackPad 2 与最新的 MacBook 上）或者 3D Touch 的确是一个为 iPhone 定制的似曾相识却不同的东西。3D Touch 允许操作系统和应用侦测每一个手指与屏幕接触时的压力。从用户体验的角度来说，最大的变化莫过于当你用点力去触碰或者拖拽屏幕时，操作系统将会触发诸如 peek，pop 这些新机制。那么问题来了：**我们是否能够在网站中使用这个新玩意呢？让我们一点点来看：**

iOS 9 搭载的 Safari 包含了一些用于 “Force Touch” 的新 API，但它们其实并不是那个用于 iPhone 6s 3D Touch 的 API。你可以理解为这些 API 就是 MacBook 版 Safari 里为 Force Touch 准备的那些 API ，因为共享一套 codebase，所以它理所当然得存在了 iOS 版里而已。

Force Touch API 为我们添加了两个新东西：

1. 你的 click 事件处理函数将会从 MouseEvent 中收到一个新的属性：`webkitForce`
2. DOM 也新增了四个事件：`(webkit)mouseforcewillbegin`，`mouseforcedown`，`mouseforceup` 与 `mouseforcechange`。下边的示意图将告诉你这些事件是在何时被触发的：

![Force Events](http://www.mobilexweb.com/wp-content/uploads/2015/09/foceevents.png)


相信你已经从它们的名字中意识到了，这些事件都是基于鼠标而非触摸的，毕竟它们是为 MacBook 设计的。并且，TouchEvent 也并没有包含 `webkitForce` 这个属性，它仅仅存在于 MouseEvent 里。在 iOS Safari 里，你确实可以找到 `onwebkitmouseforce` 这一系列事件处理器，但是很可惜它们并不会被触发，click 返回的 MouseEvent 也永远只能得到一个 `webkitForce: 0`


可喜可贺的是，故事还没有结束。[Touch Events v2 draft spec（触摸事件第二版草案）](https://w3c.github.io/touch-events/) 中正式添加了 `force` 属性。3D Touch 也得以在 iPhone 6s 与 6s+ 中通过 TouchEvent 访问到。不过，笔者也要在这里提醒大家，由于没有 `webkitmouseforcechange` 这样给力的事件，在手机上我们只能通过 **轮询 TouchEvent 的做法** 来不断检测压力值的改变……非常坑爹

[@Marcel Freinbichler](https://twitter.com/fr3ino) 第一个在 Twitter 上晒出了自己的 [Demo](http://freinbichler.me/apps/3dtouch)。在 6s 或 new Macbook 的 Safari（目前仅 Safari 支持）上访问就可以看到圆圈会随着压力放大。墙内的小伙伴可以直接试试下面这个圆圈，体验下 3D/Force Touch 带来的的奇妙体验。

<iframe src="//huangxuan.me/forcify/" style="
    width:100%;
    height:500px;
    border: 0;
"></iframe>

如果你不巧在用不支持 3D/Force Touch 的设备，发现尼玛用力按下去之后居然圆圈也有反映！？

放心，这真的不是你的设备突然习得了“感应压力”这项技能，而是因为 [Forcify](http://huangxuan.me/forcify) 是一个用于在所有设备上 polyfill 3D/Force Touch API 的 JS 库……它不但封装了 OSX/iOS 两个平台之间 API 的差异，还使用"长按"来模拟了 `force` 值的变化……



### iPad Pro

全新的 iPad Pro（12.9 寸）打破了以往 iPad 渲染网站的方式。在此之前，市面上所有的 iPad（从初代 iPad，到 iPad Air 4，到 iPad Mini）都是以 768px 的宽度提供 viewport。

而屏幕更大的 iPad Pro 选择了宽 1024px 的 viewport，这使得它天生就能容纳更多的内容。不少人说iPad Pro 就是抄 Microsft Surface Pro 的嘛……嗯哼，IE/Edge 在 Surface Pro 上就是以 1024px 作为视口宽度的……

从交互的角度上来说，iPad Pro 虽然不支持 3D Touch，但是可以搭配 Smart Keyboard 与/或 Apple Pen（带有压力侦测）使用。对于键盘其实并没有什么好说的，如果一个网站在搭配键盘的桌面电脑上好用，它在 iPad Pro 上应该也不赖。而对于 Apple Pen，很可惜，目前似乎并没有 API 能让你在网站上获得这根笔的压力与角度。


## 新的 iOS 操作系统特性

### iPad 上的多任务处理

自 iOS 9 起，iPad 允许两个应用在同一时刻并肩执行，有三种方式：**Slide Over**，**Split View** 与 **Picture-in-Picture**。不过，每一种方式都有其硬件需求，比如说 Slide Over 需要 iPad Air, iPad Mini 2 以上的设备，而 Split View 由于对内存的要求目前只支持 iPad Air 2 与 iPad Pro。

#### Slide Over（滑过来！）

Slide Over 支持的 App 并不多，不过 Safari 名列其中，这意味着我们的网站将可能在这个模式下被渲染。当网站处于 Slide Over 模式下时，它将在屏幕的右 1/4 位置渲染，并且置于其他 native app 之上。

这个模式也为 Responsive Web Design（响应式网站设计）提出了新的挑战：**一个只为 iPad 优化的网站，也需要能在该设备上以无需手动刷新的形式支持小屏幕的渲染。**因此，如果你正在使用服务器端探测（RESS），那么你的 iPad 版本需要以某种方式包含手机版本的网站，或者在进入该模式后重新加载一次。（如果你不了解 RESS，你可以观看我的[另一篇博文](/2014/11/20/responsive-web-design/)）


![Slide Over](http://www.mobilexweb.com/wp-content/uploads/2015/09/slideover.png)

在这个模式下，无论横屏还是竖屏，所有的 iPad（包括 Pro）都会把你的网站以 320px 的 viewport 宽度进行渲染，就好像在一个大 iPhone 5 上一样。你可以在 CSS 中通过 media query（媒体查询）探测到这个模式：

```css
/* iPad Air or iPad Mini */
(device-width: 768px) and (width: 320px)
/* iPad Pro */
(device-width: 1024px) and (width: 320px)
```

#### Split View（分屏视图）

在较新版本的 iPad 上，你可以将 Slide Over 的 Side View（侧视图）升级为 Split View。此时，两个应用将以相同比例在你的屏幕上同时工作。

在这个模式下，我们的网站将可能……

* **以屏幕 1/3 比例渲染时**，viewport 在 iPad Air/mini 犹如 iPhone 5，宽 320px。而在 iPad Pro 上则像是 iPhone 6：宽 375px
* **以屏幕 1/2 比例渲染时**，viewport 在 iPad Air/mini 上呈现为 507px 宽，而在 iPad Pro（横屏）下呈现为 678px 宽
* **以屏幕 2/3 比例渲染时**，viewport 在 iPad Air/mini 上呈现为 694px 宽，而在 iPad Pro（横屏）下呈现为 981px 宽

![Split View](http://www.mobilexweb.com/wp-content/uploads/2015/09/splitview.png)


#### Picture in Picture（画中画）

在一些较新版本的 iPad 上，使用 HTML5 video 标签的网站可以将其暴露到 Picture in Picture 机制中。通过 API（本文稍后会讲）或用户的触发，视频可以独立于网站在其他应用的上方继续播放。

![Picture in Picture](http://www.mobilexweb.com/wp-content/uploads/2015/09/pip.png)


### iOS 9 下的响应式网页设计

下图向你展示了 iOS 9 所有可能的 viewport 尺寸，检查检查你的响应式断点都包含它们了吗？

![iOS 9 RWD](http://www.mobilexweb.com/wp-content/uploads/2015/09/ios9rwd.png)

### Safari View Controller 

如果你用过 Twitter 或者 Facebook（或者微信，微博……），那么你一定知道很多 native app 在打开一个网页链接时并不会默认使用 Safari。它们试图让你留在它们的应用里，所以通过提供 webview 让你在应用内进行网页浏览。可是问题在于，这类 webview 并不会与浏览器共享 cookies，sessions，autofill（自动填充）与 bookmark（书签），为了解决这些问题，就有了 Safari View Controller。

现在，native app 可以使用 Safari View Controller 来打开网站，它提供与 Safari 完全一致的隐私政策、local storage，cookies、sessions 同时让用户留在你的 app 中，它通过一个 “Done”（完成）按钮使用户可以回到 native app 的上一个 controller。这个全新的 controller 还可以让我们在 Share（分享）按钮上添加自定义的操作，这些操作在用户使用 Safari 应用时并不会出现。同时，native app 对这个自定义 Safari 实例具有完全的内容控制，你可以屏蔽不想被渲染的内容。

当你需要基于 web 的鉴权，比如 OAuth 时，使用 Safari View Controller 同样是一个好主意，这样就不再需要打开浏览器再重定向回你的应用。不过注意了，Safari View Controller 只适用于在线、公开的 web 内容。如果你的 web 内容假设在本地或者私服，那么 WKWebView 仍然是最推荐的选择。

> 笔者八卦一下，Safari View Controller 实际上也算是半个社区推进的产物。早在 2014 年 12 月，Tumblr 的 iOS 工程师 Bryan 就发表了一篇著名的 [We need a “Safari view controller”](http://bryan.io/post/104845880796/we-need-a-safari-view-controller) 叙述现有 webview 在第三方登录鉴权时的窘境。
> 2015 年 6 月，Apple Safari 工程师 Ricky Mondello 的 Twitter 宣告了这个设想的落地：You all asked for it. Come see me introduce it. Introducing Safari View Controller 1:30 PM, Tuesday. Nob Hill.


### Safari Content Blockers

现在，iOS 9 上的 Safari 支持一种全新的 App Extensions（应用拓展）：**Content Blocker**（内容拦截器）。这类拓展以 native app 的形式存在，你可以在 App Store 上下载到，它们可以拦截 Safari 内的任何内容，包括：跟踪器、广告、自定义字体、大图片、JavaScript 文件等等。

作为 web 开发者，尽管我们不能禁用 Content Blocker，我们仍然应该注意到它们的存在。诸如 Crystal 的一些拦截器宣称他们[可以提高网页的打开速度](http://murphyapps.co/blog/2015/8/22/crystal-benchmarks)。Crystal 声称可以加快网页的加载速度 3.9 倍并且少用 53% 的带宽。不过问题是：到底哪些东西被拦截器拦截了？[这篇文章](http://thenextweb.com/apple/2015/08/27/content-blocking-in-ios-9-is-going-to-screw-up-way-more-than-just-ads/)提到了一些我们未来可能会遇到的问题。

![crystal](http://www.mobilexweb.com/wp-content/uploads/2015/09/crystal.png)

在 iOS 9 发布后，Peace，一个 Content Blocker，曾在 App Store 排名跻身前十。从用户的角度来说，如果一个网站由于被 Content Blocker 拦截了某些重要资源而不能正常工作，你可以长按重新加载按钮并且以不启用 Content Blocker 的方式重新加载这个网站（见下图，来自 MacWorld.com）

![disable content blocker](http://www.mobilexweb.com/wp-content/uploads/2015/09/macworld.png)

Content Blocker 能隐藏元素，也有能力通过 CSS 选择器、域名、类型、或者 URL 来过滤并拦截某个文件的加载，[Purify Blocker](https://itunes.apple.com/us/app/purify-blocker-fast-clutter/id1030156203?ls=1&mt=8) 给用户提供了拦截某一种内容类型的进阶选项，比如 Web Fonts。

![purify](http://www.mobilexweb.com/wp-content/uploads/2015/09/purify.png)

### WKWebView 的增强

UIWebView 已经被官方弃用，虽然它还在在那，不过它再也不会得到什么升级。与此相反，WKWebView 正在取代它的位置。一个最受期待的特性现在终于推出：加载本地文件到 WKWebView。因此，现在 Apache Cordova 应用与其他 web 内容都可以直接从 iOS 包中使用本地文件了，不再需要各种诡异的 hack 了。

此外，还有一些新特性也一并推出。比如说，通过 WKWebsiteDataStore，Objective-C 或 Swift 有能力查询与管理 webview 的本地存储（比如 localStorage 或 IndexedDB）。这就允许我们将原有的数据存储替换成新的某些东西，比如说替换成一个不永久的（Chrome for iOS 的隐身模式就需要这种东西）

### Universal Links（通用链接）

如果你既有一个网站，又有一个 native app，你现在可以通过 Universal Links 来增强用户体验。它允许你在操作系统内“占有”自己的域名，这样，一切指向你网站的链接都会被重定向到你的 app。

目前，所有的 app 都是通过自定义 URI 来达到这个效果的，比如 `comgooglemaps://` 就可以用来从网站或者其他原生 iOS 应用中打开 Google Maps。

想要提供这个特性的话，你首先需要在 native app 中实现 Deep Linking（深度链接），让应用中的内容与 Safari 的 URL 吻合。然后，你需要在 Apple 的网站上关联你的域名，取得这个域名的 SSL 认证并且把签名后的 JSON 部署到该域名上。这是为了防止第三方的应用“占据”了属于你而不属于他们的域名，比如说 twitter.com 被非 Twitter 的其他应用占据掉。

目前唯一的缺点是用户好像并不能决定到底以哪种方式来打开内容（使用 web 还是 app），不过我们可以观望一段时间看看它会如何发展。在不远的这段时间里，你可能会发现在网站或 Google 搜索里点击一个链接时会没有任何预警的就跳进了 native app 里。

### App Search（应用搜索）

Apple 带着自己的 web 蜘蛛杀进了搜索的市场，而我们需要支持它得以在 Siri 与 Spotlight 中提升自己的曝光率。这在我们同时拥有网站与 app 时尤为重要，因为现在 Apple 会索引你网站的内容，但打开时却可能将用户带到了 app 里去。

尽管这会开启 SEO 的新篇章，不过却相当容易。你需要使用一些标签标准，诸如 [Web Schema](http://schema.org/)、[AppLinks](http://applinks.org)、[OpenGraph](http://ogp.me) 或者 [Twitter Cards](https://dev.twitter.com/cards/mobile)，配合上 App Banner 与 `app-argument`，如果你有你自己的 native app 的话。

> 关于“让你的网页支持 Apple 搜索”的更多详情，请查阅 Apple 官方文档 [Mark Up Web Content](https://developer.apple.com/library/prerelease/ios/documentation/General/Conceptual/AppSearch/WebContent.html#//apple_ref/doc/uid/TP40016308-CH8-SW5)

Apple 刚刚发布了一个 [App Search Validation Tool（应用搜索验证工具）](https://search.developer.apple.com/appsearch-validation-tool/)来帮助你搞清楚，需要向你的网站添加什么才能支持 App Search

![App Search](http://www.mobilexweb.com/wp-content/uploads/2015/09/appsearch-1024x467.png)

### CloudKit JS

如果你拥有一个 native app，你很可能会将用户数据保存在 iCloud 上。在过去，只有 iOS 与 Mac 应用被允许使用它。现在，通过 CloudKit JS，你的网站也可以连接上 iCloud 数据了。

### Back Button

现在，当你链接到一个 native app 时（通过自定义 URI 或者 Universal Link），Safari 会询问用户是否想要使用 native app 打开这个链接（见下图）。如果用户同意了，这个应用将被打开，并且在左上角会有一个返回按钮可以返回 Safari ，返回到你的网站。

<img src="http://www.mobilexweb.com/wp-content/uploads/2015/09/back.png" alt="backbutton" width="320" />

## 新的 API 支持

### Navigation Timing API

Navigation Timing API 在 iOS 9 迎来了回归。让我们回忆一下，这货添加于 8.0 却在一周后的 8.1 中去掉了。这对于 Web 性能是个好消息。通过这个 API，我们可以更精确的测量时间，还可以获得一系列有关加载过程的时间戳，它们对于追踪与在真实场景中做决策来改进用户体验都非常有用。


### Picture in Picture

PiP API（被称为 Presentation Mode API）目前只支持 iOS，它允许我们手动让一个 `<video>` 元素进入或离开 PiP 模式如果 `video.webkitSupportsPresentationMode` 是支持的。

举个例子，我们可以在内嵌模式与 PiP 模式中切换：

```js
video.webkitSetPresentationMode(
    video.webkitPresentationMode === "picture-in-picture" ?
    "inline" : 
    "picture-in-picture"
);
```

我们还可以通过新的 `onwebkitpresentationmodechanged` 事件来检测 Presentation Mode（展示模式）的变化。


### Backdrop CSS

iOS 7 与最近的 Mac OS 使用 Backdrop filter（背景滤镜）来模糊背景（指 native 开发），而在网站上实现这个却并不容易。

iOS 9 上的 Safari 现在支持了来自 Filter Effect v2 spec（滤镜特效第二版规范）的 **backdrop-filter**。比如说，我们可以使用一个半透明的背景并且对其背后的背景使用滤镜：

```css
header {
   background-color: rgba(255, 255, 255, 0.4);
   -webkit-backdrop-filter: blur(5px);
   backdrop-filter: blur(5px);
}
```

![backdrop](http://www.mobilexweb.com/wp-content/uploads/2015/09/backdrop.png)


### CSS Scroll Snapping

在 web 上实现分页内容（比如相册跑马灯）总是非常麻烦，无论是使用 JavaScript 框架、touch 事件还是 hacking 滚动条等等。Apple 新添加了一个很赞的 CSS 特性叫做 CSS Scroll Snapping。这个特性新增了一系列的 CSS 属性让你定义规则或者不规则的 snap zone（停留区域），这样滚动的位置就会“啪”地一下停在这个区域，而非像以前一样可以停在任何地方。  

来看个例子：

```css
#photo-gallery{
    width: 100%;
    overflow-x: scroll;
    -webkit-scroll-snap-points-x: repeat(100%);
    -webkit-scroll-snap-type: mandatory;
}
```

> 想要看个跑起来后的例子？笔者为大家准备了 webkit 的官方 [demo](http://www.webkit.org/demos/scroll-snap/)，不过这个属性目前只支持 iOS 9 Safari 哦，并不支持 webview


### CSS Supports

CSS Supports，包括 CSS `@supports` 与来自 CSS Conditional Rules Module Level 3 spec 的 JavaScript CSS Supports API 都在 iOS 上迎来降临。现在，我们可以针对某个 CSS 属性的特定值的支持情况来编写代码：

```css
@supports(-webkit-scroll-snap-type: mandatory) {
    /* we use it */
}
```

同样，使用 JavaScript：

```js
if (CSS.supports("-webkit-scroll-snap-type", "mandatory")) {}
```

### 一些细微的改进

* ECMAScript 6 的更完善支持：classed、computed properties、template literial 与 week sets
* 新的 CSS Level4 伪类/元素选择器：`:not`、`:matches`、`:any-link`、`:placeholder-shown`、`:read-write`、`:read-only`
* Native app 现在可以通过 extension 来向 Safari 的 Shared Links（分享链接）窗口上注入信息
* 大量无前缀 CSS 属性的支持（终于），比如 transition、animation、@keyframes、flex 与 columns
* Mac OS El Capitán 上的 Safari 9 提供了一个全新设计的 Web Inspector（Web 检查器）。幸运的是，iOS 9 的远程调试完全兼容 Mac OS 上的 Safari 8，所以你倒是不用急着升级你的 Mac OS
* iOS 9 通过 `-apple-font` 加入了一些 Dynamic Fonts（动态字体），并且它们现在应用的是 Apple 的新字体：San Francisco，笔者的博客就已经用上它啦
* scrollingElement 现在可用了
* `<input type=file>` 现在允许你从 iCloud Drive 与已安装的第三方应用，比如 Google Drive 中选择文件

<img src="http://www.mobilexweb.com/wp-content/uploads/2015/09/IMG_2017.png" alt="input file" width="320" />

* 当你加载一个 HTTPS 协议的页面时，你不能混用 HTTP 与 HTTPS 的资源


## Bugs

Bug 通常都要在几周之后才会显露出来，我也会持续跟进并更新这篇文章。目前为止，我的发现如下：

* 对于 Home Screen webapps（添加至主屏的 web 应用），`apple-mobile-web-app-status-bar-style` 这个 meta 标签不起作用了！所以你现在不能再像过去一样使用 `black-translucent` 让你的 webapp 渲染在状态栏的后面了。（iOS 9.2 fixed 了这个 bug）
* Speech Synthesis API （语音综合 API）不再工作了


## 仍在等待……

当 Mac 上的 Safari、桌面电脑与 Android 上的 Chrome 都已经为网站支持 Push Notification （通知推送）时，iOS 上的 Safari 仍然不支持这个特性。就 API 而言，我们仍然没有：WebRTC、getUserMedia、Service Worker、FileSystem API、Network Information API、Battery Status API、Vibration API 等等……你又在 iOS 上等待哪些特性呢？ 

## watchOS 与 tvOS

新发布的 watchOS 2.0 与 tvOS 9.0 都是基于 iOS 的操作系统，它们针对特定的设备发行（Apple Watch 与新的 Apple TV）。从用户的角度来说，那里并没有浏览器了。从开发者的角度，那里也没有 Webview 了。

尽管有不少人抱怨（大部分都是针对 webview 的缺失），我并不能确定这是不是个坏主意。我猜测 Apple 会尝试通过 Siri 来将 “web” 带给 TV、手表、甚至 CarPlay 的用户。所以，如果你遵循了上述的 “App Search” 的步骤，你的内容将可能通过 Siri 在这些设备上以 widget（小部件）或者快捷回复的形式变得可以访问。

对于 Apple TV ，它支持使用 JavaScript、DOM API 与 XMLHttpRequest 来让我们构建某种类似 Client-Server webapp 的东西。没有 HTML 和 CSS，这是什么把戏？其实它支持的叫 TVML，是一种基于 XML、为那些可以被渲染在 TV 屏幕上的特定内容而优化后的标签。这些标签只可以在来自应用商店的 native app 中渲染，但是这些 TVML 是由服务器端来生成的。


## 著作权声明

本文译自 [iOS 9, Safari and the Web: 3D Touch, new Responsive Web Design, Native integration and HTML5 APIs --- Breaking the Mobile Web](http://www.mobilexweb.com/blog/ios9-safari-for-web-developers)   
译者 [黄玄](http://weibo.com/huxpro)，首次发布于 [Hux Blog](http://huangxuan.me)，转载请保留以上链接