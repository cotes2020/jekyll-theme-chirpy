---
title: "CSS3动画设计 - 字体特效"
date: 2019-07-16
permalink: /2019-07-16-font-animation/
categories: ["开源技术课程", "CSS3 动画设计"]
---
## 特效一览


**划线动态**：


![css-1.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-16-font-animation/6d3f1787ab0c48306073b6859180a039.gif)


**背景高亮**：


![css2.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-16-font-animation/ca5f4f1781712992fe24b2c734b32535.gif)


**色块进出**：


![css3.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-16-font-animation/74ce3777bda6a66a12bbb87da4f4fc5b.gif)


## 划线动态


### 原理


首先，利用`::after`和`::before`就可以画出上下两条线，所以只需要一个 dom 元素即可。


其次，对于鼠标移入的动画，要给上面两个伪元素设置`:hover`选择器。


最后是处理动画方向。我们以上面的线条为例，在鼠标移入的时候，是从右到左变化的。这里是通过设置`transform-origin`属性来修改动画方向。下面的线条同理，方向相反即可。


**注意**：代码是通过`scaleX`来实现缩放，相比于设置`width`，会启用 GPU，避免重绘。


### 代码


html 代码：


```html
<body>
  <span>dongyuanxin.github.io</span>
</body>
```


css 代码：


```css
span {
  color: #595959;
  position: relative;
  z-index: 1;
}

span::before,
span::after {
  content: "";
  z-index: -1;
  position: absolute;
  left: 0;
  right: 0;
  height: 2px;
  background: #262626;
  transform: scaleX(0);
  transition: transform 0.2s ease-in-out;
}

span::before {
  top: 0;
  transform-origin: center right;
}

span::after {
  bottom: 0;
  transform-origin: center left;
}

span:hover {
  cursor: pointer;
}

span:hover::before {
  transform-origin: center left;
  transform: scaleX(1);
}

span:hover::after {
  transform-origin: center right;
  transform: scaleX(1);
}
```


## 背景高亮


### 原理


首先，利用`::before`伪元素就可以模拟出覆盖需要的色块。所以仅仅需要一个 dom 元素。这里伪元素的`content`元素必须给，否则不会显示（有些坑）。


其次，色块大小改变是通过`scaleY`来设置的，原因和第一个动画原因一样。


最后，伪元素的色块会覆盖 dom 上的元素。所以需要给 dom 元素设置`z-index`，并且让其生效并大于伪元素的`z-index`。


### 代码


html 代码：


```html
<body>
  <span>dongyuanxin.github.io</span>
</body>
```


css 代码：


```css
span {
  color: #d9d9d9;
  position: relative;
  z-index: 1;
}

/*
1. content必须给
2. 用transform覆盖 配合 z-index
*/
span::before {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  left: -0.25em;
  right: -0.25em;
  z-index: -1;
  background: #262626;
  transform: scaleY(0.2);
  transform-origin: center bottom;
  transition: all 0.1s linear;
}

span:hover {
  cursor: pointer;
}

span:hover::before {
  transform: scaleY(1);
}
```


## 色块进出


### 原理


这和上一个“背景高亮”动画类似，不同的是色块的位置和大小变化方向不同。其余基本一致。


### 代码


html:


```html
<body>
  <span>dongyuanxin.github.io</span>
</body>

```


css:


```css
span {
  color: #d9d9d9;
  position: relative;
  z-index: 1;
}

span::before {
  content: "";
  z-index: -1;
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
  background: #262626;
  transform-origin: center right;
  transform: scaleX(0);
  transition: transform 0.1s linear;
  /* 这里不要指明为 all */
}

span:hover {
  cursor: pointer;
}

span:hover::before {
  transform-origin: center left;
  transform: scaleX(1);
}

```


