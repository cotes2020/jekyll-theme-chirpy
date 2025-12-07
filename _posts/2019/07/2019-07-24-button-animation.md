---
title: "CSS3动画设计 - 按钮特效"
date: 2019-07-24
permalink: /2019-07-24-button-animation/
categories: ["开源技术课程", "CSS3 动画设计"]
---
## 特效一览


**滑箱**：


![css3-1.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-24-button-animation/7502824d8c7b71396dfb3c872cbbe595.gif)


**果冻**：


![css3-2.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-24-button-animation/7652f58fcf5ecf982b58de73931a7fa1.gif)


**脉冲**：


![css3-3.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-24-button-animation/ac623b115a1cdf24dd9ced33cf0629f9.gif)


**闪光**：


![css3-4.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-24-button-animation/6e0b99cbdca1270ed058edb8fe86e02d.gif)


**气泡**：


![css3-5.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-24-button-animation/931869337e4bf2f6ef3bb083a6ff1f63.gif)


## 滑箱特效


### 原理


因为 button 元素可以使用 before/after 伪元素，所以借助伪元素，可以实现动态图中的遮盖层。


为了避免回流重绘，滑箱的运动方向是垂直方向，所以使用`scaleY`属性。对于动画的方向，需要借助`transform-origin`改变动画原点。


### 代码实现


html：


```html
<button>xin-tan.com</button>
```


css：


```css
button {
  outline: none;
  border: none;
  z-index: 1;
  position: relative;
  color: white;
  background: #40a9ff;
  padding: 0.5em 1em;
}

button::before {
  content: "";
  z-index: -1;
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
  background-color: #fa541c;
  transform-origin: center bottom;
  transform: scaleY(0);
  transition: transform 0.4s ease-in-out;
}

button:hover {
  cursor: pointer;
}

button:hover::before {
  transform-origin: center top;
  transform: scaleY(1);
}
```


## 果冻特效


### 原理和代码


果冻特效可以分割成 5 个部分，所以无法简单通过 `transition` 来实现，要借助`animation`。并且动画触发的时间点是鼠标移入的时候，因此 `animation` 要在`:hvoer`中声明。


```css
button {
  z-index: 1;
  color: white;
  background: #40a9ff;
  outline: none;
  border: none;
  padding: 0.5em 1em;
}

button:hover {
  cursor: pointer;
  animation: jelly 0.5s;
}
```


下面开始编写 jelly 动画的特效。这个动画可以分解为 4 个部分：「初始 => 挤高 => 压扁 => 回到初始状态」。挤高 和 压扁这里都是通过`scale`来实现的，代码如下：


```css
@keyframes jelly {
  0%,
  100% {
    transform: scale(1, 1);
  }

  33% {
    transform: scale(0.9, 1.1);
  }

  66% {
    transform: scale(1.1, 0.9);
  }
}
```


### 更进一步


上面的动态已经仿真不错了，如果将 4 部分变成 5 部分：「初始 => 挤高 => 压扁 => 挤高 => 回到初始状态」。**视觉上会有一种弹簧的特效**，就像手压果冻后的效果：


```css
@keyframes jelly {
  0%,
  100% {
    transform: scale(1, 1);
  }

  25%,
  75% {
    transform: scale(0.9, 1.1);
  }

  50% {
    transform: scale(1.1, 0.9);
  }
}
```


## 脉冲特效


### 原理和代码


首先，还是去掉 button 的默认样式。注意设置 button 的`z-index`属性并且让其生效，要保证其大于 `::before` 的 `z-index` 属性，**防止 dom 元素被伪元素覆盖**。


```css
button {
  position: relative;
  z-index: 1;
  border: none;
  outline: none;
  padding: 0.5em 1em;
  color: white;
  background-color: #1890ff;
}

button:hover {
  cursor: pointer;
}
```


剩下的就是设置伪元素。因为脉冲特效给人的感觉是“镂空”放大。因此，变化对象是 `border` 属性。而镂空的效果，是通过透明背景来实现的。


```css
button::before {
  content: "";
  position: absolute;
  z-index: -1;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
  border: 4px solid #1890ff;
  transform: scale(1);
  transform-origin: center;
}
```


动画启动时间是鼠标移入，border 上变化的是颜色变淡和大小变小，透明度也逐渐变成 0。


```css
button:hover::before {
  transition: all 0.75s ease-out;
  border: 1px solid#e6f7ff;
  transform: scale(1.25);
  opacity: 0;
}
```


⚠️ transition 和 transform 是放在`hover`状态下的伪元素，目的是让动画瞬间回到初始状态。


## 闪光特效


### 原理和代码


实现上依然是借助伪元素，闪光特效更多注重的是配色，动画方面实现的核心是利用`rotate`来实现「倾斜」的效果，利用`translate3d`来实现「闪动」的效果。


```css
button {
  outline: none;
  border: none;
  z-index: 1;
  position: relative;
  color: white;
  background: #262626;
  padding: 0.5em 1em;
  overflow: hidden;
  --shine-width: 1.25em;
}

button::after {
  content: "";
  z-index: -1;
  position: absolute;
  background: #595959;
  /* 核心代码：位置一步步调整 */
  top: -50%;
  left: 0%;
  bottom: -50%;
  width: 1.25em;
  transform: translate3d(-200%, 0, 0) rotate(35deg);
  /*  */
}

button:hover {
  cursor: pointer;
}

button:hover::after {
  transition: transform 0.5s ease-in-out;
  transform: translate3d(500%, 0, 0) rotate(35deg);
}
```


⚠️`translate3d`除了避免重绘回流，还能启用 GPU 加速，性能更高。但之前为了方便讲述，一般使用的是`translate`属性。


## 气泡特效


### 原理和代码


首先，还是禁用 button 元素的默认样式，并且调整一下配色：


```css
button {
  outline: none;
  border: none;
  cursor: pointer;
  color: white;
  position: relative;
  padding: 0.5em 1em;
  background-color: #40a9ff;
}
```


由于 button 的伪元素层级是覆盖 button 的，所以要设置 `z-index` 属性，防止伪元素遮盖显示。毕竟只想要背景色的遮盖，字体不需要遮盖。在上面的样式中添加：


```css
button {
  z-index: 1;
  overflow: hidden;
}
```


最后处理的是伪元素的变化效果。**特效是从中心向四周蔓延**，所以应该让其居中。


对于大小变化，还是利用`scale`属性。


因为是圆形，所以将`border-radius`设置为 50%即可。


```css
button::before {
  z-index: -1;
  content: "";
  position: absolute;
  top: 50%;
  left: 50%;
  width: 1em;
  height: 1em;
  border-radius: 50%;
  background-color: #9254de;
  transform-origin: center;
  transform: translate3d(-50%, -50%, 0) scale(0, 0);
  transition: transform 0.45s ease-in-out;
}

button:hover::before {
  transform: translate3d(-50%, -50%, 0) scale(15, 15);
}
```


### 换个方向？


示例代码中的气泡特效是从中间向四周扩散，如果想要从左上角向右下角扩散呢？例如下图所示：


![css3-6.gif](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-07-24-button-animation/20e8d9d2fa1ee35e9bb327e5bf56d282.gif)


处理过程很简单，**只需要改变一下气泡的初始位置即可**。


```css
button::before {
  z-index: -1;
  content: "";
  position: absolute;
  width: 1em;
  height: 1em;
  border-radius: 50%;
  background-color: #9254de;
  /* 变化位置的代码 */
  top: 0;
  left: 0;
  transform-origin: center;
  transform: scale3d(0, 0, 0);
  transition: transform 0.45s ease-in-out;
  /* *********** */
}

button:hover::before {
  transform: scale3d(15, 15, 15);
}
```


## 参考链接

- [《transform-origin: 改变动画原点》](http://caibaojian.com/transform-origin.html)
- [《Increase Your Site’s Performance with Hardware-Accelerated CSS》](https://blog.teamtreehouse.com/increase-your-sites-performance-with-hardware-accelerated-css)
- [《css3 变量》](https://www.ruanyifeng.com/blog/2017/05/css-variables.html)

