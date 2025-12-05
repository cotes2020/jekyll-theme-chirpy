---
title: "CSS3动画设计 - Loader特效基础篇"
url: "2019-07-25-loader-animation-first"
date: 2019-07-25
---

## 特效一览


**声音波纹**：


![css4-1.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-25-loader-animation-first/5cfd57f064c17852af8747ba3e19a409.gif)


**弹性缩放**：


![css4-2.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-25-loader-animation-first/0b720b3b6c544e26300b83fc7e7db285.gif)


**旋转加载**：


![css4-3.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-25-loader-animation-first/61230a15ba93fc89716e0ea6b6d14e8a.gif)


**渐变点**:


![css4-4.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-25-loader-animation-first/f5a4cee502593cba6ccd859732d36399.gif)


**翻转纸片**：


![css4-5.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-25-loader-animation-first/9c978fab128510ed82ee8204610efe5a.gif)


## 声音波纹特效


### 原理和代码


需要几个块，就准备几个空 dom 元素。当然，数量越多，动画越细腻，但同时维护成本也高。


```html
<div>
    <span></span>
    <span></span>
    <span></span>
    <span></span>
</div>
```


先编写一些基础样式代码：


```css
div {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 2em;
}

span {
    width: 0.3em;
    height: 1em;
    background: red;
}
```


单独观察一个空 dom 元素，其实就是一个缩放动画：


```css
@keyframes grow {
    0%,
    100% {
        transform: scale3d(1, 1, 1);
    }

    50% {
        transform: scale3d(1, 2, 1);
    }
}
```


**不同点**：每个元素的动画启动时间不一样；每个元素的初始状态不一样。


下面的代码中有个 2 个时间参数，第一个是动画时长，第二个是延迟时间。如果延迟时间是负数，那么会自动计算对应时间点的动画作为初始状态动画。


```css
div span:nth-of-type(1) {
    animation: grow 1s 0s ease-in-out infinite;
}

div span:nth-of-type(2) {
    animation: grow 1s 0.15s ease-in-out infinite;
}

div span:nth-of-type(3) {
    animation: grow 1s 0.3s ease-in-out infinite;
}

div span:nth-of-type(4) {
    animation: grow 1s 0.45s ease-in-out infinite;
}
```


## 弹性缩放特效


### 原理和代码


分解一下这个动画，会发现它分为 2 个部分：

1. 放大一倍，旋转 360deg
2. 缩小到正常大小，再旋转 360deg

因此，动画的代码如下：


```css
@keyframes stretch {
    0% {
        transform: scale(1) rotate(0);
    }

    50% {
        transform: scale(2) rotate(360deg);
    }

    100% {
        transform: scale(1) rotate(720deg);
    }
}
```


样式效果是通过 border 来实现的，只展示对立方向的 border 即可：


```css
div {
    width: 1em;
    height: 1em;
    border: 2px transparent solid;
    border-top-color: #531dab;
    border-bottom-color: #531dab;
    border-radius: 50%;
    animation: stretch 2s ease-in-out infinite;
}
```


## 缩放加载特效


### 原理和代码


动画很简单，就是无限循环的旋转。主要是如此丝滑的效果，采用的是「慢 => 快 => 慢」对应的动画函数，也就是`ease-in-out`。


样式效果和上一个类似，也是通过操作 border 实现。


```css
div {
    height: 1em;
    width: 1em;
    border: 2px solid #d3adf7;
    border-radius: 50%;
    border-top-color: #722ed1;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}
```


## 渐变点特效


### 原理和代码


和「声音波纹」特效类似，展示点的个数取决于空 dom 元素个数：


```html
<div>
    <span></span>
    <span></span>
    <span></span>
</div>
```


动画特效非常简单，就是「透明 => 完全不透明 => 透明」这个过程。整体效果也是通过调整动画时长 && 动画延迟启动时间来实现的。


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
    animation: fade 1s ease-in-out infinite;
}

div span:nth-of-type(2) {
    animation: fade 1s 0.2s ease-in-out infinite;
}

div span:nth-of-type(3) {
    animation: fade 1s 0.4s ease-in-out infinite;
}

@keyframes fade {
    0%,
    100% {
        opacity: 0;
    }

    50% {
        opacity: 1;
    }
}
```


## 翻转纸片


### 代码和原理


这个特效比较有意思的地方是**动画很细腻**。主要是分为两部分，一个是关于 y 轴的 180deg 旋转，另一个是关于 x 轴的 180deg 旋转。要借助`rotateX`和`rotateY`来实现。


⚠️ 为了取得位于中间位置的轴线，要设置`transform-origin: center`。


代码如下：


```css
div {
    width: 24px;
    height: 24px;
    background: #eb2f96;
    transform-origin: center;
    animation: paper 2s ease infinite;
}

@keyframes paper {
    0% {
        transform: rotateX(0) rotateY(0);
    }

    50% {
        transform: rotateX(180deg) rotateY(0);
    }

    100% {
        transform: rotateX(180deg) rotateY(180deg);
    }
}
```


