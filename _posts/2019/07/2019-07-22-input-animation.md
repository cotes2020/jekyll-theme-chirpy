---
title: "CSS3动画设计 - 输入框特效"
date: 2019-07-22
permalink: /2019-07-22-input-animation/
categories: ["开源技术课程", "CSS3 动画设计"]
---
## 特效一览


**划线动态**：


![css2-1.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-22-input-animation/da4c521fd9e9801d37270ab5dfd2b40f.gif)


**动态边框**：


![css2-2.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-22-input-animation/2e73d9765d8b37d9e7ce1a2d11089227.gif)


## 划线动态


### 原理和代码


`:before` 和 `:after`伪元素指定了一个元素文档树内容之前和之后的内容。由于`input`标签不是可插入内容的容器。所以这里下划线无法通过伪元素来实现。需要借助其他 dom 节点。


```html
<div>
  <input type="text" />
  <span></span>
</div>

```


包裹在外的父元素`div`应该设置成`inline-block`，否则宽度会满屏。


```css
div {
  position: relative;
  display: inline-block;
}
```


`input` 标签需要禁用默认样式：


```css
input {
  outline: none;
  border: none;
  background: #fafafa;
}
```


`span`标签实现「左进右出」的动态，需要改变`transform-origin`方向。为了避免回流重绘，通过`scaleX`来实现宽度变化的视觉效果。


```css
input ~ span {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: 1px;
  background-color: #262626;
  transform: scaleX(0);
  transform-origin: right center;
  transition: transform 0.3s ease-in-out;
}

input:focus ~ span {
  transform: scaleX(1);
  transform-origin: left center;
}
```


## 动态边框


### 原理和代码


如动态图所示，有 4 条边框。所以除了`input`元素外，还需要准备其他 4 个 dom。为了方便定位，嵌套一个父级元素。


```html
<div>
  <input type="text">
  <span class="bottom"></span>
  <span class="right"></span>
  <span class="top"></span>
  <span>
</div>
```


和「划线动态」类似，input 和 div 的样式基本一样。为了好看，改一下 padding 属性。


```css
div {
  position: relative;
  display: inline-block;
  padding: 3px;
}

input {
  outline: none;
  border: none;
  background: #fafafa;
  padding: 3px;
}
```


对于其他 4 个 span 元素，它们的位置属性，动画属性，以及颜色都是相同的：


```text
.bottom,
.top,
.left,
.right {
  position: absolute;
  background-color: #262626;
  transition: transform 0.1s ease-in-out;
}

```


对于.bottom 和.top，它们的变化方向是水平；对于.left 和.right，它们的变化方向是垂直。


```css
.bottom,
.top {
  left: 0;
  right: 0;
  height: 1px;
  transform: scaleX(0);
}

.left,
.right {
  top: 0;
  bottom: 0;
  width: 1px;
  transform: scaleY(0);
}
```


下面就是处理延时的特效。动态图中，动画按照下、右、上、左的顺序依次变化。借助的是`transition-delay`属性，来实现动画延迟。


```css
.bottom {
  bottom: 0;
  transform-origin: right center;
}
input:focus ~ .bottom {
  transform: scaleX(1);
  transform-origin: left center;
}

.top {
  top: 0;
  transform-origin: left center;
  transition-delay: 0.2s;
}
input:focus ~ .top {
  transform: scaleX(1);
  transform-origin: right center;
}

.right {
  transform-origin: top center;
  right: 0;
  transition-delay: 0.1s;
}
input:focus ~ .right {
  transform: scaleY(1);
  transform-origin: bottom center;
}

.left {
  left: 0;
  transform-origin: bottom center;
  transition-delay: 0.3s;
}
input:focus ~ .left {
  transform: scaleY(1);
  transform-origin: top center;
}
```


## 参考链接

- [为什么 input 不支持伪元素(:after,:before)？](https://www.zhihu.com/question/21296044)

