---
title: "[译]SCSS和CSS3对比"
date: 2019-05-17
permalink: /2019-05-17-why-i-stopped-using-sass/
---
## 翻译说明


这是一篇介绍现代 css 核心特性的文章，并且借助 sass 进行横向对比，充分体现了 css 作为一门设计语言的快速发展以及新特性为我们开发者带来的强大生产力。


第一次尝试翻译技术文，为了让文章更通俗易懂，很多地方结合了文章本意和自己的说话风格。另外，时间有限水平有限，难免有些失误或者翻译不恰当的地方，欢迎指出讨论。


**英文原文地址**：[https://cathydutton.co.uk/posts/why-i-stopped-using-sass/](https://cathydutton.co.uk/posts/why-i-stopped-using-sass/)


## 正文开始


我每年都要重新搭建和设计我的网站，这是一个非常不错的方式去跟进 HTML/CSS 的最新进展、开发模式和网站生成器。在上个月，我发布了新版本：从 Jekyll 和 GithubPages 迁移到 Eleventy 和 Netlify。


一开始，我并没有移除代码中所有的 sass 代码。这本不是我计划中的事情，但随着我不断查看 sass 代码，我一直在思考：它们是否给网站带来了价值，还是仅仅增加了复杂度和依赖性(特指对：scss)？随着这年 css 的发展，曾经让我使用 sass 的原因似乎不那么重要了。


其中一个例子就是我已经移除了媒体查询。当我了解到 CSS 的一些新的特性，那些针对特定屏幕大小的代码（媒体查询）没有必要，因此被移除了。


## Sass 解决了什么问题？


大概 5、6 年前，我第一次了解到 sass 的时候，我是有些换衣的。随着我搭建越来越多的响应式 web 应用，我才意识到借助 sass 的  `functions`  和  `mixins`  可以大大提高代码复用。显而易见的是，随着设备、视图窗口和主题等场景的变化，使用（sass 的）变量让代码迁移的成本更低。


下面是我用 sass 做的事情：

- 布局
- 变量
- Typography

## 布局


布局一直是 css 中让人困惑的地方。而响应式布局正是我最初决定使用 Sass 去创建 css 布局的重要原因。


### 使用 sass


我一直记得我第一次尝试用 css 创建一个响应式网格布局的时候，那要为每列创建一个对应的类名，然后再用语义化不强的类名（比如  `col-span-1`  和  `col-span-4` ）来标记它。


```css
.col-span-3 {
    float: left;
    width: 24%;
    margin-left: 1%;
}
.col-span-4 {
    float: left;
    width: 32.3%;
    margin-left: 1%;
}
.col-span-5 {
    float: left;
    width: 40.6%;
    margin-left: 1%;
}
```


借助 sass 的  `mixin`  和变量，能够不再编写像上面那样的类名。并且能够通过改变  `$gridColumns`  变量，来创造更灵活的布局。


下面是我写的第一个基于  `mixin`  的网格布局：


```scss
@mixin grid($colSpan, $gridColumns: 12, $margin: 1%) {
    $unitWidth: $gridColumns / $colSpan;
    float: left;
    width: (100 - $unitWidth * $margin) / $unitWidth;
    margin: 0 $margin/2;
}
```


引入方法如下：


```scss
.sidebar {
    @include grid(3);
}
.main-content {
    @include grid(9);
}
@media only screen and (max-width: 480px) {
    .sidebar {
        @include grid(12);
    }
    .main-content {
        @include grid(12);
    }
}
```


### 使用 CSS


通过 css 的  `grid`  的介绍，我们不再需要用语义化不强的类名或者 sass 或者其他预处理器，来完成网格布局这项功能。Rachel Andrew 说这种方法是最好的：


> 你不再需要一种工具来帮助你创建网格布局，因为你现在就拥有它。


下面的的代码基于内容的宽度范围，创建了一个响应式布局，并且不再需要预设和计算设备的大小。


```scss
.project {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(12em, 1fr));
    grid-gap: 1em;
}
```


从 sass 创建网格布局转变为 css 原生网格布局，是一个“无痛”体验。它不仅仅能够减少对 sass 的依赖，还可以让我编写更灵活的代码，激发更多的设计思路以及不再使用媒体查询设计网站。


但是最明显的不足是浏览器的兼容性。Grid 是目前只在最新浏览器中被支持，包括 IE11、IE10。对  `auto-fill`  和  `auto-fit`  属性的支持更少，但可以通过查询规范支持来提前规避。


## 变量


变量就是一个可能变化的值，我一直不知道 css 中有这个功能。今天我的大多数项目都遵循  [ITCSS methodology](https://cathydutton.co.uk/posts/why-i-stopped-using-sass/) ，并且创建一个配置文件专门用来存放变量定义。通常，我会为字体样式、颜色和媒体查询设置变量。


之前 sass 的做法：


```scss
/* COLORS */
$colors: (
    "black": #2a2a2a,
    "white": #fff,
    "grey-light": #ccc7c3,
    "grey-dark": #2a2a2a,
    "accent": #ffa600,
    "off-white": #f3f3f3,
    "sky-blue": #ccf2ff
);
/* BREAKPOINTS */
$breakpoints: (
    "break-mobile": 290px,
    "break-phablet": 480px,
    "break-tablet": 768px,
    "break-desktop": 1020px,
    "break-wide": 1280px
);
/* TYPOGRAPHY */
$font-stack: (
    decorative: #{"oswald",
    Helvetica,
    sans-serif},
    general: #{"Helvetica Neue",
    Helvetica,
    Arial,
    sans-serif}
);
```


使用变量或者映射让我的网站能够快速和简单地应对大的改动。它也预防了在大型代码项目中过分堆积复杂的外形、颜色变量，特别是 hover 悬浮的动画、引用、边框等等。


例如下面场景：


```scss
.button {
    background-color: #4caf50; /* Green */
}
.button:hover {
    background-color: #3f8c42; /* Dark Green */
}
.button:active {
    background-color: #266528; /* Darker Green */
}
```


能够被 sass 的变量和颜色相关的内置函数重写：


```scss
$button-colour: #4caf50;
.button {
    background-color: $button-colour;
}
.button:hover {
    background-color: darken($button-colour, 20%);
}
.button:active {
    background-color: darken($button-colour, 50%);
}
```


### 到底有什么不同？


css 自带的变量能做的事情更多，不仅仅是替换静态字面量，它可以实时动态计算（而不仅仅是编译构建的时候静态替换）。它允许被 js 修改，并且不需要在代码外面再包裹一层  `mixins`  和  `funtions` 。


```scss
:root {
    --button-color: #4caf50;
}
.button {
    background-color: var(--button-color);
}
header .button {
    --button-color: #000000;
    background-color: var(--button-color);
}
```


当然，sass 中对颜色的一些内置函数在 css 中也可以使用：


```scss
:root {
    --button-color: #4caf50;
}
.button:hover {
    color: color-mod(var(--button-color) tint(50%));
}
```


不幸的是，这（颜色相关内置函数）一直在处在提案阶段。我决定还是手动定义颜色变量来替换它（提案中的方案）。


```scss
.button {
    background: var(--colour-dark);
}
.button:hover {
    background: var(--colour-bright);
    text-decoration: underline;
}
```


如果你执意使用他们，那么这个包含了很多 css 颜色函数功能的  [PostCSS](https://github.com/jonathantneal/postcss-color-mod-function)  项目能够帮助到你。


## 网页排版


最后，对于排版，在之前的代码中，我是用 sass 去创建响应式排版和布局。下面展示的  `mixin`  的用法让我能轻易地处理不同大小的屏幕与设备：


```scss
@mixin typography($size) {
    font-size: $size;
    @include mq(break-desktop) {
        font-size: $size * 1.2;
    }
}
```


现在，我用原生的 css 的功能来进行这些计算：


```scss
:root {
    --font-size: calc(18px + 0.25vw);
}
body {
    font-size: var(--font-size);
}
```


## 展望


CSS 正在朝向更具内涵的规范发展，在 css 的  `grid`  特性中，有  `flexbox`  以及  `min-content` 、 `max-content` 、 `fit-content`  这些属性，而在 Css Grid Layout Module Level2 中也准备加入的新布局： `Subgrid` 。


这些新的特性都让原生的 css 更有吸引力！


## 更多资料

- [Sass: @mixin 指令介绍](https://www.jianshu.com/p/00200dd89131): 相遇比`@extend`继承，`@mixin`更灵活

