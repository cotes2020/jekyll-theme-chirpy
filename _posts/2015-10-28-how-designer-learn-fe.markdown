---
layout:     post
title:      "设计师如何学习前端？"
subtitle:   "How designers learn front-end development?"
date:       2015-10-28 12:00:00
author:     "Hux"
header-img: "img/home-bg-o.jpg"
tags:
    - 知乎
    - Web
    - UX/UI
---

> 这篇文章转载自[我在知乎上的回答](https://www.zhihu.com/question/21921588/answer/69680480)，也被刊登于[优秀网页设计](http://www.uisdc.com/head-first-front-end)等多个网站上 ;)


笔者的经历在知乎就可以看到，大学专业是数字媒体艺术，大一实习过动效设计师，大二拿到了人生第一个大公司 offer 是阿里的交互设计，后来转岗到淘宝旅行的前端团队，现在在微信电影票做前端研发。
<br>
<br>也是走过了不少野路子，不过还好有小右哥 <a data-hash="cfdec6226ece879d2571fbc274372e9f" href="//www.zhihu.com/people/cfdec6226ece879d2571fbc274372e9f" class="member_mention" data-editable="true" data-title="@尤雨溪" data-tip="p$b$cfdec6226ece879d2571fbc274372e9f">@尤雨溪</a> 这样艺术/设计转前端的大神在前面做典范，也证明这条路是玩的通的 ;)
<br>
<br>接下来就说说自己的学习建议吧，一个小教程，也是自己走过的流程，仅供参考哈
<br>
<br>------------
<br>
<br><b>背景篇</b>
<br>
<br>在这个时代学习新东西，一定要善于使用 Bing/Google 等搜索引擎…网络上的资源非常丰富，自学能力也尤为重要，尤其是对于学习技术！
<br>
<br>
<br>
<br><b>入门篇（HTML/CSS）</b>
<br>
<br>说起设计师希望学前端的初衷，大概还是因为各种华丽的网页特效/交互太过吸引人，这种感觉大概就是：“Hey，我的设计可以做成网页访问了呢！”
<br>好在，“展示”对于前端技术来说反而是最简单的部分。所以，放下你对“编程”两个字的恐惧，<b>从“称不上是编程语言”的 HTML/CSS 开始，先做点有成就感的东西出来吧！</b>
<br>
<br>对于设计师来说，最有成就感的一定是“可以看到的东西”，而 HTML/CSS 正是用来干这个的，HTML 就是一堆非常简单的标签，而 CSS 无非就是把你画画的流程用<b>英语</b>按一定的格式写出来而已：
<br>


```html
<p> p is paragraph! </p>

<style>
p { color: red;}
</style>
```


是不是非常容易，就跟读英语一样！
<br>接下来，你就需要开始自学啦，比如常用 HTML 标签的意思，各种 CSS 的属性，还有 CSS 的盒模型、优先级、选择器……放心，它们都很容易；能玩得转 PS/AI/Flash/Axure/AE/Sketch 的设计师们，学这个洒洒水啦
<br>
<br>推荐几个资源：
<br>
<ul>
    <li><a href="//link.zhihu.com/?target=http%3A//www.w3school.com.cn/" class=" wrap external" target="_blank" rel="nofollow noreferrer">w3school 在线教程<i class="icon-external"></i></a> (中文，一个很 Low 但是又很好的入门学习网站）
        <br>
    </li>
    <li><a href="//link.zhihu.com/?target=http%3A//www.codecademy.com/" class=" wrap external" target="_blank" rel="nofollow noreferrer">Learn to code<i class="icon-external"></i></a> (Codecademy，如果你英文 OK，<b>强烈建议</b>你使用它进行交互式的学习！里面从 HTML/CSS 到搭建网站的课程都有，免费，生动直观）
        <br>
    </li>
</ul>
<br><b>这个阶段的练习主要是“临摹”：用代码画出你想画的网站，越多越好。</b>
<br>
<br>对于书，我<b>非常不推荐</b>上来就去看各种厚厚的入门/指南书，没必要！这一个阶段应该快速上手，培养兴趣，培养成就感。先做出可以看的东西再说，掌握常用的 HTML/CSS 就够用了
<br>
<br>如果完成的好，这个阶段过后你大概就可以写出一些简单又好看的“静态网页”了，比如这个作品集/简历：<a href="//link.zhihu.com/?target=http%3A//huangxuan.me/portfolio/" class=" wrap external" target="_blank" rel="nofollow noreferrer">Portfolio - 黄玄的博客<i class="icon-external"></i></a> （好久没更新了…丢人现眼）
<br>
<br>
<br>
<br><b>入门篇（JavaScript/jQuery）</b>
<br>
<br>想要在网页上实现一些交互效果，比如轮播图、点击按钮后播放动画？那你就必须要开始学习 JavaScript 了！JavaScript 是一门完整、强大并且非常热门的编程语言，你在浏览器里看到的所有交互或者高级功能都是由它在背后支撑的！
<br>
<br>举个小栗子：
<br>

```js
alert("Hello World!")
```

就这一行，就可以在浏览器里弹出 Hello World 啦！
<br>
<br>在了解一些基础的 JavaScript 概念（变量、函数、基本类型）后，我们可以直接去学习 jQuery，你不用知道它具体是什么（它是一个 JavaScript 代码库），你只要知道它可以显著地降低你编写交互的难度就好了：
<br>

```js
$('.className').click(function(){
  alert("Hello jQuery")
})
```

通过 jQuery，我们可以继续使用在 CSS 中学到的“选择器”
<br>
<br>对于没有编程基础的人来说，想要完全掌握它们两并不容易。作为设计师，很多时候我们可以先不必深究它们的原理，而是尝试直接应用它！这样成就感会来得很快，并且你可以通过实际应用更加理解 JavaScript 是用来做什么的。
<br>
<br>我仍然推荐你使用 <a href="http://www.w3school.com.cn/" target="_blank" rel="nofollow noreferrer">w3school 在线教程</a> 与 <a href="//www.codecademy.com/" target="_blank" >http://www.codecademy.com/</a> 进行学习。另外，你可以看一看诸如《<a href="//link.zhihu.com/?target=http%3A//book.douban.com/subject/10792216/" class=" wrap external" target="_blank" rel="nofollow noreferrer">锋利的jQuery (豆瓣)<i class="icon-external"></i></a>》 这一类非常实用的书籍，可以让你很快上手做出一些简单的效果来！
<br>
<br>如果学习得顺利，你还可以尝试使用各种丰富的 jQuery 插件，你会发现写出支持用户交互的网站也没有那么困难～很多看上去很复杂的功能（比如轮播图、灯箱、下拉菜单），搜一搜然后看看文档（教程）、改改示例代码就好了。
<br>
<br>比如说，配合 <a href="//link.zhihu.com/?target=https%3A//github.com/Huxpro/jquery.HSlider" class=" wrap external" target="_blank" rel="nofollow noreferrer">Huxpro/jquery.HSlider · GitHub<i class="icon-external"></i></a> 这样的轮播图插件，你可以很轻松的写出 <a href="//link.zhihu.com/?target=http%3A//huangxuan.me/jquery.HSlider/" class=" wrap external" target="_blank" rel="nofollow noreferrer">HSlider | Demo<i class="icon-external"></i></a> 这样的网页相册或者 <a href="//link.zhihu.com/?target=http%3A//huangxuan.me/jquery.HSlider/demo-weather-app/" class=" wrap external" target="_blank" rel="nofollow noreferrer">HSlider | Weather<i class="icon-external"></i></a> 这样的手机端 App 原型～
<br>
<br>最后，我想推荐下 <a href="//link.zhihu.com/?target=http%3A//getbootstrap.com/" class=" wrap external" target="_blank" rel="nofollow noreferrer">Bootstrap · The world's most popular mobile-first and respons<i class="icon-external"></i></a> ，这是世界上最知名的前端 UI 框架之一，提供了大量 CSS 样式与 jQuery 插件。它非常容易学习并且中英文教程都非常健全，你并不需要理解它背后的工作原理就能很好的使用它，让你快速达到“可以建站的水平”。有余力的话，你不但可以学习如何使用它，还可以学习它背后的设计思想。
<br>
<br>
<br>
<br><b>转职方向一：前端重构 （Web Rebuild）</b>
<br>
<br>业内通常把专精 HTML/CSS 的前端从业人员称为重构，而对于注重视觉效果的设计师来说，在掌握基本的 HTML/CSS 后，就可以朝着这个方向发展了。
<br>
<br><b>到了这个阶段，你不但要知道怎么写页面，还要知道它们都是为什么，并且知道怎么做更好。这对你理解 Web 世界非常有帮助，并且能帮助你做出更“系统化”的设计。</b>
<br>
<br>CSS 的学问很多，你需要开始理解文档流、浮动流等各种定位的方式与原理，理解 CSS 的继承复用思想、理解浏览器的差异、兼容、优雅降级……这里强烈推荐一本书：《<a href="//link.zhihu.com/?target=http%3A//book.douban.com/subject/4736167/" class=" wrap external" target="_blank" rel="nofollow noreferrer">精通CSS（第2版） (豆瓣)<i class="icon-external"></i></a>》，虽然前端技术突飞猛进，但这本书的思想永远不会过时。
<br>
<br>HTML 方面，要开始注重语义化、可访问性与结构的合理，你要开始学习“结构与样式的分离”，这里有一本神书将这种分离做到了极致：《<a href="//link.zhihu.com/?target=http%3A//book.douban.com/subject/2052176/" class=" wrap external" target="_blank" rel="nofollow noreferrer">CSS禅意花园 (豆瓣)<i class="icon-external"></i></a>》
<br>
<br>另外，各种炫酷屌的 CSS 3 属性你一定会喜欢：你可以用媒体查询做响应式网页设计，你可以用 transiton 和 animation 做补间动画与关键帧动画，用 transform 做缩放、旋转、3D变换，还有圆角、渐变、阴影、弹性盒！样样都是设计师的神器！
<br>
<br>如果你还掌握了 <b>入门篇（JavaScript/jQuery）</b>的知识，那么<b>恭喜你！你已经可以做出很多有趣的网页了！</b>很多 minisite 或者微信上的“H5” 小广告，这个程度的你已经可以轻松完成了！
<br>
<br>配合上你的设计功力，你可以开始尝试创作一些好玩的东西，比如这种富含交互和动画的网站 <a href="//link.zhihu.com/?target=http%3A//huangxuan.me/senova/" class=" wrap external" target="_blank" rel="nofollow noreferrer">绅宝 SENOVA<i class="icon-external"></i></a> ，它仍然是基于 <a href="//link.zhihu.com/?target=https%3A//github.com/Huxpro/jquery.HSlider" class=" wrap external" target="_blank" rel="nofollow noreferrer">Huxpro/jquery.HSlider · GitHub<i class="icon-external"></i></a> 实现的！或者给自己做个小小的个人网站试试
<br>
<br>
<br>
<br><b>转职方向二：前端工程师（Front-end Engineer）</b>
<br>
<br>如果你觉得上述的这些都还满足不了你，你渴望做出更多了不起的交互，甚至你已经喜欢上了编程，想要转行做工程师，或者成为一名全栈设计师，那么你可以朝着这个方向继续发展！
<br>
<br>这个阶段的最大难度，是你必须<b>学会像一名软件工程师一样思考</b>。你需要踏踏实实学习编程语言，深入理解作用域、对象、类、封装、继承、面向对象编程、事件侦听、事件冒泡等一大堆编程概念，你还需要了解浏览器，学习 DOM、BOM、CSSOM 的 API，你甚至还需要学习一些网络原理，包括域名、URL、DNS、HTTP 请求都是什么…
<br>
<br>你可能会被这一大堆名词吓到。确实，想要搞定他们并不容易。但是，你要相信只要你肯花功夫它们也没有那么难，而更重要的是，如果你能拿下他们，你所收获的并不只是这些而已，而是真正跨过了一道大坎 —— <b>你的世界将因此打开， 你看待世界的方式将因此改变</b>
<br>
<br>对于这个阶段，你可以继续在 <a href="//www.codecademy.com/" target="_blank" >http://www.codecademy.com/</a> 上学习，但是 w3school 已经不够用了，遇到不会的语法，我推荐你查阅 <a href="//link.zhihu.com/?target=https%3A//developer.mozilla.org/zh-CN/" class=" wrap external" target="_blank" rel="nofollow noreferrer">Mozilla 开发者网络<i class="icon-external"></i></a>，这是少数中英文都有的非常专业且友好的网站。
<br>
<br>同时，你可能需要看一些书本来帮助你学习 JavaScript ：
<br>
<ul>
    <li> 《<a href="//link.zhihu.com/?target=http%3A//book.douban.com/subject/10546125/" class=" wrap external" target="_blank" rel="nofollow noreferrer">JavaScript高级程序设计（第3版） (豆瓣)<i class="icon-external"></i></a> 》或 《<a href="//link.zhihu.com/?target=http%3A//book.douban.com/subject/2228378/" class=" wrap external" target="_blank" rel="nofollow noreferrer">JavaScript权威指南 (豆瓣)<i class="icon-external"></i></a>》，大而全的书只需要一本就够了</li>
    <li>如果上面这本你觉得太难，你可以先看 《<a href="//link.zhihu.com/?target=http%3A//book.douban.com/subject/6038371/" class=" wrap external" target="_blank" rel="nofollow noreferrer">JavaScript DOM编程艺术 （第2版） (豆瓣)<i class="icon-external"></i></a>》来过渡一下，这本书比较容易，它会教给你 “优雅降级、渐进增强”的优秀思想</li>
</ul>
<br>如果你能顺利得渡过了这个阶段，我想你已经能做出很多令你自豪的网站了！试着向身边的工程师朋友询问如何购买域名、配置简单的静态服务器，或者搜搜“Github Pages”，然后把你的作品挂在网络上让大家欣赏吧！
<br>
<br>你还可以试着用 JavaScript 写写小游戏，这不但能锻炼你的编程水平还非常有趣～比如这是我刚学 JS 不久后 hack 一晚的产物 —— 用 DOM 实现的打飞机：<a href="//link.zhihu.com/?target=http%3A//huangxuan.me/aircraft" class=" wrap external" target="_blank" rel="nofollow noreferrer">Hux - Aircraft<i class="icon-external"></i></a> （不支持手机）
<br>
<br>
<br>
<br><b>入行篇</b>
<br>
<br>如果你能完成上述所有的学习，你已经是一名非常出色的前端学徒了！对于只是想要丰富技能的设计师或者产品经理来说，接下来的内容可能会让你感到不适 ;(
<br>但如果你铁了心想要真正入行进入大公司从事专职前端开发的工作，那么你可以接着往下看：
<br>
<br>近几年的前端技术发展迅猛，前端工程师早已不是切切图写写页面做点特效就完事的职位，你需要具备相当完善的工程师素质与计算机知识，成为一名真正的工程师。
<br>
<br><b>你需要非常了解 JavaScript 这门语言</b>，包括 闭包、IIFE、this、prototype 及一些底层实现（ES、VO、AO）、熟悉常用的设计模式与 JavaScript 范式（比如实现类与私有属性）。另外，新的 ES6 已经问世，包括 class, module, arrow function 等等
<br>
<br><b>你需要非常了解前端常用的网络及后端知识</b>，包括 Ajax、JSON、HTTP 请求、GET/POST 差异、RESTful、URL hash/query、webSocket、常用的跨域方式（JSONP/CORS、HTTP 强缓存/协商缓存，以及如何利用 CDN 、静态网站/动态网站区别、服务器端渲染/前端渲染区别等等
<br>
<br><b>你需要学习使用进阶的 CSS</b>，包括熟悉 CSS 3，使用 Scss/Less 等编译到 CSS 的语言，使用 autoprefixer 等 PostCSS 工具，了解 CSS 在 Scope/Namespace 上的缺陷，你还可以学习 CSS Modules、CSS in JS 这些有趣的新玩意
<br>
<br><b>你需要非常了解前端的模块化规范</b>，可能在你学习到这里的时候，Require.js/AMD 已经再见了，但是 CommonJS 与 ES6 Modules 你必须要了解。（你可以观看我的分享《<a href="//link.zhihu.com/?target=http%3A//huangxuan.me/js-module-7day/%23/" class=" wrap external" target="_blank" rel="nofollow noreferrer">JavaScript Modularization Seven Day<i class="icon-external"></i></a>》 来学习 JS 模块化的历史）
<br>
<br><b>你需要熟悉 Git 与 Shell 的使用</b>，包括基于 git 的版本管理、分支管理与团队协作，包括简单的 Linux/Unix 命令、你要知道大部分程序员的工作可以通过 shell 更快更酷的完成，并且很多“软件”只能通过 shell 来使用。你还可以把你的代码放到 github 上与人分享，并且学习 github 上其他优秀的开源代码
<br>
<br><b>你需要熟悉并且习惯使用 Node</b>，包括了解 npm、使用 Grunt/Gulp/Browserify/Webpack 优化你的工作流、对你的代码进行打包、混淆、压缩、发布，你还可以使用 Express/Koa 配合 MongoDB/Redis 涉足到后端领域，或者尝试用 Node 做后端渲染优化你的首屏体验
<br>
<br><b>你需要了解各种 HTML 5 的新 API</b>，包括 &lt;video&gt;/&lt;audio&gt;，包括 Canvas，webGL、File API、App Cache、localStorage、IndexedDB、Drag &amp; Drop、更高级的 DOM API、Fetch API 等等
<br>
<br><b>你需要学习 JavaScript 的单线程与异步编程方法</b>，因为它们非常非常常用、包括 setTimeout/setInterval，回调与回调地狱、事件与event loop、还有 Promise 甚至 Async/Await
<br>
<br><b>你需要非常了解浏览器</b>，包括主流浏览器的名称、内核与差异、包括私有属性与 -webkit- 等厂商前缀，你需要学习如何使用 Chrome DevTool，你需要了解浏览器渲染的 reflow/repaint 来避免 Jank 并进行有针对性的性能优化
<br>
<br><b>你需要专门学习 Mobile Web</b>，因为移动互联网是趋势。包括 viewport、CSS pixel、 touch 事件、iOS/Android 浏览器的差异与兼容、移动端的性能优化、300ms delay 等等…你还需要知道 Hybrid 是什么，包括 Cordova/Phonegap，更复杂的比如和 iOS/Android 通信的机制，比如 URI Scheme 或者 JS Bridge
<br>
<br><b>你需要学习一些</b><b>非常火热的前端框架/库</b>，他们不但能帮助你更快的进行开发、更重要的是他们背后所蕴含的思想。包括 Backbone、Angular、Vue、React、Polymer 等等、了解它们背后的双向数据绑定、单向数据流、MVC/MVVM/Flux 思想、Web Component 与组件化等等
<br>
<br><b>你需要学习如何构建 web 单页应用</b>，这是 web 的未来，包括利用 history API 或者 hash 实现路由，包括基于 Ajax + 模版引擎或者其他技术的前端渲染、包括组织较为复杂的软件设计等等
<br>
<br><b>我还建议你学习更多的计算机知识</b>，它们能对你的代码能起到潜移默化的作用，包括简单的计算机体系结构、更广泛的编程知识（面向对象/函数式等）、栈、堆、数组、队列、哈希表、树、图等数据结构、时间复杂度与空间复杂度以及简单的算法等等
<br>
<br><b>你需要了解业内的大神并阅读它们的博客/知乎/微博</b>，比如 <a data-hash="cfdec6226ece879d2571fbc274372e9f" href="//www.zhihu.com/people/cfdec6226ece879d2571fbc274372e9f" class="member_mention" data-editable="true" data-title="@尤雨溪" data-tip="p$b$cfdec6226ece879d2571fbc274372e9f">@尤雨溪</a><a data-hash="3ec3b166992a5a90a1083945d2490d38" href="//www.zhihu.com/people/3ec3b166992a5a90a1083945d2490d38" class="member_mention" data-editable="true" data-title="@贺师俊" data-tip="p$b$3ec3b166992a5a90a1083945d2490d38">@贺师俊</a><a data-hash="3212f9044005e9306aab1b61e74e7ae6" href="//www.zhihu.com/people/3212f9044005e9306aab1b61e74e7ae6" class="member_mention" data-editable="true" data-title="@张云龙" data-tip="p$b$3212f9044005e9306aab1b61e74e7ae6">@张云龙</a><a data-hash="c5198d4e9c0145aee04dd53cc6590edd" href="//www.zhihu.com/people/c5198d4e9c0145aee04dd53cc6590edd" class="member_mention" data-editable="true" data-title="@徐飞" data-tip="p$b$c5198d4e9c0145aee04dd53cc6590edd">@徐飞</a><a data-hash="20fdd386a6e59d178b8fe14e2863cb40" href="//www.zhihu.com/people/20fdd386a6e59d178b8fe14e2863cb40" class="member_mention" data-editable="true" data-title="@张克军" data-tip="p$b$20fdd386a6e59d178b8fe14e2863cb40">@张克军</a><a data-hash="c11336b8607d86bc9090bed90757a34c" href="//www.zhihu.com/people/c11336b8607d86bc9090bed90757a34c" class="member_mention" data-editable="true" data-title="@玉伯" data-tip="p$b$c11336b8607d86bc9090bed90757a34c">@玉伯</a><a data-hash="64458d15a75902cd0425732b7b757705" href="//www.zhihu.com/people/64458d15a75902cd0425732b7b757705" class="member_mention" data-editable="true" data-title="@拔赤" data-tip="p$b$64458d15a75902cd0425732b7b757705">@拔赤</a><a data-hash="0d9b98af12015c94cff646a6fc0773b5" href="//www.zhihu.com/people/0d9b98af12015c94cff646a6fc0773b5" class="member_mention" data-editable="true" data-title="@寸志" data-tip="p$b$0d9b98af12015c94cff646a6fc0773b5">@寸志</a><a data-hash="790dccce26904cdcd11b0fad3bac37b7" href="//www.zhihu.com/people/790dccce26904cdcd11b0fad3bac37b7" class="member_mention" data-editable="true" data-title="@题叶" data-tip="p$b$790dccce26904cdcd11b0fad3bac37b7">@题叶</a><a data-hash="85de6407f2219137df29b4249b91cfd5" href="//www.zhihu.com/people/85de6407f2219137df29b4249b91cfd5" class="member_mention" data-editable="true" data-title="@郭达峰" data-tip="p$b$85de6407f2219137df29b4249b91cfd5">@郭达峰</a> 等等等等，很多思想和新东西只有从他们身上才能学到。我还推荐你多参加技术交流会，多认识一些可以一起学习的小伙伴，你们可以互相交流并且一起成长
<br>
<br><b>你需要具备很强的自学能力、对技术有热情并且不断跟进</b>。因为 JavaScript/前端的社区非常非常活跃，有太多的新东西需要你自己来发现与学习：比如 Universal JavaScript、Isomorphic JavaScript、前端测试、HTML5 页游、WebRTC、WebSocket、CSS 4、SVG、HTTP/2、ES 7、React Native、Babel、TypeScript、Electron 等等等等…
<br>
<br>
<br>虽然一下扯得有点多，但这些确实就是你未来将会遇到的。你并不需要全部掌握它们，但是却多多益善；你也可以专精在某几个方面，这已经足以让你成为非常专业的前端工程师。
<br>
<br><b>所以，如果你自认为涵盖了上述要求的 40%，欢迎简历发 huangxuan@wepiao.com ，实习/全职皆可～</b>
<br>
<br>
<br>咦，这个结尾怪怪的……
