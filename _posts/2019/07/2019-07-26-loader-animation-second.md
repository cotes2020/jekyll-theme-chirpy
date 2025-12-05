---
title: "CSS3动画设计 - Loader特效·进阶篇"
url: "2019-07-26-loader-animation-second"
date: 2019-07-26
---

## 特效一览


🌊 波浪特效：


![css5-1.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-26-loader-animation-second/5326df16cde46c46f75051fe7700c6b0.gif)


🕙 撞钟特效：


![css5-2.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-26-loader-animation-second/336acfaff4a0128a57839676f4bc87f2.gif)


⏳ 沙漏特效：


![css5-3.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-26-loader-animation-second/8722cec5fa0f74aa3d52d0a208416e84.gif)


🏃 追逐特效：


![css5-4.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-26-loader-animation-second/a01f91b985c381560945397c819f9729.gif)


## 🌊 波浪特效


### 原理和代码


这里的动画效果是分成 2 个过程：上 => 下 => 回到上。**其实这两个过程是相反的**。可以使用动画属性`alternate`，在奇数次数（1、3、5 等等）正常播放，而在偶数次数（2、4、6 等等）向后播放。


```css
div {
    width: 3.5em;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

div span {
    width: 1em;
    height: 1em;
    border-radius: 50%;
    background: red;
    transform: translateY(0);
    animation: wave 1.2s ease-in-out alternate infinite;
}

div span:nth-of-type(2) {
    animation-delay: -0.2s;
}

div span:nth-of-type(3) {
    animation-delay: -0.4s;
}

@keyframes wave {
    from {
        transform: translateY(-100%);
    }
    to {
        transform: translateY(100%);
    }
}
```


⚠️ 在不清楚`alternate`之前，有尝试过将`wave`过程拆分成 2 部分。但是这样动画函数`ease-in-out`是作用于整个过程，而不是作用于其中一个过程。动画的观感上就不再具有「波浪律动」的效果。


## 🕙 撞钟特效


### 原理和代码


准备 3 个 dom 元素，左起第一个和第三个有动画特效，第二个没有。


```css
div {
    display: flex;
    position: absolute;
    align-items: center;
    justify-content: center;
}

div span {
    height: 10px;
    width: 10px;
    background: #ff4d4f;
    border-radius: 50%;
}

div span:nth-of-type(1) {
    animation: left 2s ease-in-out infinite;
}

div span:nth-of-type(2) {
    margin: 0 1px;
}

div span:nth-of-type(3) {
    animation: right 2s ease-in-out infinite;
}
```


对于这两个动画特效，乍一看是使用了延迟启动。但是延迟启动无法实现，因为只有动画第一次启动时候延迟，当动画重复开始的时候并不会延迟。**因此需要在动画过程中，让其有一段时间处于静止状态**。


```css
/* 0 ~ 50% 移动；50% ～ 100%静止 */
@keyframes left {
    0%,
    50% {
        transform: translateX(0);
    }
    25% {
        transform: translateX(-100%);
    }
}

/* 0 ~ 50% 静止；50% ～ 100%移动 */
@keyframes right {
    0%,
    50% {
        transform: translateX(0);
    }
    75% {
        transform: translateX(100%);
    }
}
```


## ⏳ 沙漏特效


### 原理和代码


沙漏特效这里仅仅需要一个`div`元素模拟容器，利用伪元素模拟里面的沙子。容器的动画是旋转；里面沙子的动画是配合旋转，在对应时刻填充 / 消失。


```css
div {
    position: relative;
    z-index: 1;
    width: 1em;
    height: 1em;
    border: 3px #d46b08 solid;
    animation: spin 1.5s ease infinite;
}

div::before {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    right: 0;
    left: 0;
    background: #fa8c16;
    transform: scaleY(1);
    transform-origin: center top;
    animation: fill 3s linear infinite;
}
```


首先来想象一下现实中旋转沙漏的效果（只考虑容器的一半），刚开始，沙漏是满的；180 度转过来后，沙子会自动到下面，此时这半部分沙漏是空的；最后再转过来，沙子又会回到这部分容器。


对于容器来说，其实就是不停的旋转；对于沙子来说，分成 2 个过程：满 => 消失 => 满。


```css
/* 容器 */
@keyframes spin {
    to {
        transform: rotate(180deg);
    }
}
/* 沙子 */
@keyframes fill {
    50% {
        transform: scaleY(0);
    }

    0%,
    100% {
        transform: scaleY(1);
    }
}
```


## SVG 特别篇：追逐特效 🏃


### 绘制 SVG


首先，我们需要绘制 svg 标签，这里绘制的是一个以(50, 50)为圆心，半径为 10 的圆形。


```html
<svg viewBox="25 25 50 50">
    <circle cx="50" cy="50" r="10" />
</svg>
```


为了方便维护，关于线条的样式均放在了样式表中编写：


```css
svg {
    width: 3.75em;
    animation: rotate 2s linear infinite;
    transform-origin: center;
}

svg circle {
    fill: none;
    stroke: red;
    stroke-width: 2;
    stroke-linecap: round;
    animation: dash 3s linear infinite;
}
```


动画分为 2 个部分，一个是旋转，一个是关于 svg 线条的变化。旋转需要指明动画方向是`center`，这个在`svg`标签设置`viewBox`时，才会生效。


```css
@keyframes rotate {
    to {
        transform-origin: center;
        transform: rotate(360deg);
    }
}
```


### stroke-dasharray 和 stroke-dashoffset


stroke-dasharray 用来指明实现、虚线的长度。比如 `stroke-dasharray: 10 30`，就是说实线和虚线长度分别为 10 和 30。如果总长度远超过 10 + 30 = 40，那么一直是 10、30、10、30......这样的循环。


stroke-dashoffset 用来指明绘制的起点。如果是正数，那么绘制起点在默认起点之前，整体有一部分被隐藏了；如果是负数，那么绘制起点在默认起点之后，整体的视觉效果是向前推进。


stroke-dashoffset 比较不容易理解，这里举个 🌰。还是以前面准备好的 svg 为例，整个调整的效果如下图所示：


![css5-5.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-26-loader-animation-second/7ec8b9d33bedf2168ca18a39a199f379.gif)


### 实现动画效果


分解一下动画的过程：「逐渐变长，并且前移 => 继续前移 => 回复到初始长度」。借助上部分所述的 stroke-dasharray 和 stroke-dashoffset，动画实现如下：


```css
@keyframes dash {
    0% {
        stroke-dasharray: 1, 100;
        stroke-dashoffset: 0;
    }
    50% {
        stroke-dasharray: 20, 100;
        stroke-dashoffset: -15;
    }
    100% {
        stroke-dasharray: 20, 100;
        stroke-dashoffset: -62;
    }
}
```


