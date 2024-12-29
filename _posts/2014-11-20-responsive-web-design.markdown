---
layout:     post
title:      "你们觉得响应式好呢，还是手机和PC端分开来写？"
date:       2014-11-20 12:00:00
author:     "Hux"
header-img: "img/post-bg-rwd.jpg"
tags:
    - 知乎
    - Web
---

> 这篇文章转载自[我在知乎上的回答](http://www.zhihu.com/question/25836425/answer/31564174)


<div>
	<p>
		<b>根据你的产品特点，进行两种不同的设计，</b>
	    <br><b>根据你的设计需求，选择合适的技术方案</b>。
    </p>
    <br><b>A与B不是硬币的正反面，它们为了解决同一个问题而生，它们是同一种思想的延伸。</b>
    <br>
    <br>
    <blockquote>移动和桌面设计的差别远不止是布局问题。只要有足够的编程量，这些差别是可以通过响应式设计来解决的。事实上，你可以认为如果一种设计不能兼顾两种平台的主要差别，就不能算是合格的响应式设计。但是，如果确实想要处理好平台间的所有差异，我们就回到了原点：进行两种不同的设计。
        <br>
        <br>——《Mobile Usability》（《贴心设计 打造高可用性的移动产品》）</blockquote>
    <br>
    <br>其实无论是什么解决方案，我们先来看看我们想要解决的问题：
    <br>
    <br><b>“屏幕尺寸越来越多，不同设备的交互特质也有着巨大的差别，我们希望我们的网站能够在移动手机、平板、桌面电脑，在键鼠、触摸、无障碍设备上都有优秀的用户体验。所以，我们需要网站的用户界面在不同的平台上有所不同。”</b>
    <br>
    <br>
    <br>那怎么做呢，一个解决方案应运而生：
    <br>
    <br>
    <ul>
        <li><b>响应式设计 (Responsive Web design)</b>
        </li>
    </ul><b>狭义上</b>，我们把<b>主要依靠前端 CSS</b> （包括 Media Query 媒体查询，百分比流式布局，网格与Typography系统……）来对各种屏幕尺寸进行响应的做法，称之为响应式布局，又称作自适应网页设计，或者弹性设计。
    <br>
    <br>这种主要依靠CSS的方案有很多优点，比如：
    <br>
    <ul>
        <ul>
            <li>设计元素很容易被复用，设计成本低</li>
            <li>前端只需要维护一套CSS代码，<b>维护成本</b>低</li>
            <li>桌面端与移动端的设计十分接近，令用户感到“熟悉”</li>
            <li>不需要任何服务器端的支持</li>
            <li>与业务耦合程度低，复用程度高（ 以至于 Bootstrap、Foundation 等一干框架都跟进了这个解决方案 ）</li>
        </ul>
    </ul>但问题也很明显，比如：
    <br>
    <ul>
        <ul>
            <li>设计需求复杂时，前端的<b>开发成本</b>没有任何减轻</li>
            <li>无论是针对桌面还是移动的CSS代码（甚至图片资源文件）都会被同等的下载到客户端（<b>没有考虑移动端的网络优化</b>）</li>
            <li>如果JS不写两套，桌面端的交互和移动端的交互很难针对平台作出差异</li>
        </ul>
    </ul>
    <br>
    <br>如果<b>你的</b><b>移动用户对网站所有的功能和内容有着与桌面用户同等的需求</b>，比如 新闻、报纸（媒体类）网站，或者活动、专题页等 <b>偏重信息传达而轻交互 </b>的网站，那么这个解决方案其实恰到好处：
    <br><b>触摸屏优化（胖手指）、减少次要信息…… 这些通过 CSS 解决就够了。</b>
    <br>
    <br>
    <br><b>但是，如果我想要做更多的 「移动化设计」，比如 减少信息层级、增强手势操作、让网页更接近一个Native App ？</b>
    <br>
    <br>好吧，为了更复杂的需求，为了我们的网站能更牛逼的 <b>「响应」</b> 各个平台，
    <br>又有了这些解决方案：
    <br>
    <br>
    <br>
    <ul>
        <li><b>服务器端（后端）：</b>
            <br>
        </li>
        <ul>
            <li>RESS （Responsive Web Design with Server Side Components）通过服务器端组件的响应式网页设计</li>
        </ul>
    </ul>提倡 RESS 的人认为：基于前端 CSS 的响应式方案只是一种妥协：
    <br><b>“ UI 只是在很被动的进行「调整」，而不能真正达到各个平台的最优。好的设计应该达到「这个设备该有的体验」（Device Experiences）。 ”</b>
    <br>
    <blockquote><b>Device Experiences ：</b>A device experience is defined by how a device is most commonly used and the technical capabilities or limitations it possesses.</blockquote>RESS 的本质还是服务器端动态的生成，返回 HTML、JS、CSS、图像等资源文件，但是只使用同一个 URL 就可以提供给移动端定制化更强的网页，同时还大大节省了网络资源。
    <br>
    <br>
    <br>
    <ul>
        <li><b>前端</b>（主要是JS），比如：
            <br>
        </li>
        <ul>
            <li>在 JavaScript 中实现两套逻辑，分别兼容键鼠、触摸设备</li>
            <li>通过 UA、特性检测 在前端做设备判断，对资源进行异步加载，渲染不同模版</li>
            <li>通过 特性检测 在前端做设备判断，使用不同的业务逻辑</li>
            <li>前端的模块化也可以帮助解决这个问题，比如针对不同的平台加载不同的模块</li>
            <li>……</li>
        </ul>
    </ul>
    <br>
    <br>这下，我们的网站可以更牛逼的 <b>“响应”</b> 各个平台了。
    <br>（对，我还是称之为响应：这的确还是在<b>“响应”</b>啊 ，不是吗？）
    <br>
    <br>
    <br><b>但是等下……</b>
    <br>后端开发成本上去了，前端开发成本也上去了，配合着估计产品、设计资源也都上去了，<b>那我们为什么不干脆把 移动设备网站 和 桌面设备网站 分开呢！？</b>
    <br>
    <br>
    <br>是啊，如果你的需求真的都到这一步了，你的移动网站也应该可以被称作 WebApp 了。<b>这种时候，把移动设备网站彻底分开或许真的是更好的选择。</b>
    <br>
    <br>开发资源如此充足，你还可以让专门的团队来维护移动端的网站。
    <br>（嗯，BAT 就是这么干的）
    <br>
    <br>于是又一个概念来了：
    <br>
    <br>
    <ul>
        <li><b>独立的移动版网站</b> （按题主的话来说：手机和PC端分开来写）</li>
    </ul>不过，它有那么独立么？
    <br>我们知道，我们访问网站是通过 URL 来访问的。
    <br>将移动网站 和 桌面网站 分开，如果不使用 RESS 技术，往往也就意味着要维护两个URL（不同的二级域名）
    <br>难道我们要让所有桌面用户自觉访问 <a href="http://taobao.com" class=" external" target="_blank" rel="nofollow noreferrer"><span class="invisible">http://</span><span class="visible">taobao.com</span><span class="invisible"></span><i class="icon-external"></i></a> ，所有 移动用户 都自觉访问 <a href="http://m.taobao.com" class=" external" target="_blank" rel="nofollow noreferrer"><span class="invisible">http://</span><span class="visible">m.taobao.com</span><span class="invisible"></span><i class="icon-external"></i></a> ？
    <br>
    <br>不可能吧 ＝ ＝。
    <br>
    <br>于是，我们还是得依靠前端或服务器端的一次 <b>“响应”</b>（设备检测），做 URL 重定向，才能将不同设备的用户带到那个为他们准备的网站。
    <br>
    <br>
    <br>
    <br><b>所以其实在我看来，手机和PC端分开来写，只是 狭义响应式设计 的一种发展和延伸罢了。他们的界限没有，也并不需要那么清晰。</b>
    <br>
    <br>就如开题所引用的：
    <br>
    <blockquote><b>事实上，你可以认为如果一种设计不能兼顾两种平台的主要差别，就不能算是合格的响应式设计。</b>
    </blockquote><b>“而无论是用什么解决方案。” —— 这句是我补的。</b>
    <br>
    <br>
    <br>
    <br>
    <br>故我的结论是：
    <br>
    <br><b>这不是一个二选一的问题，而是选择一个合适的度</b>（你的桌面版本代码与移动版本代码分离、耦合的程度）
    <br>
    <br>而这个度，则是由你的设计需求决定的。
    <br>而我们的需求原点其实也很简单：
    <br>
    <br> “<b>根据你的产品特点，进行两种不同的设计</b>”。
    <br>
    <br>
    <br>以上。
    <br>
    <br>
</div>
