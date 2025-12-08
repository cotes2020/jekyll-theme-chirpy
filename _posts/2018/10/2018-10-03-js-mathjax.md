---
title: "MathJax：让前端支持数学公式"
date: 2018-10-03
permalink: /2018-10-03-js-mathjax/
categories: ["实战分享"]
---

## 1. 必须要说

### 1.1 开发背景

博主使用`Vue`开发的[个人博客](https://godbmw.com/)，博文使用`markdown`语法编写，然后交给前端渲染。为了更方便的进行说明和讲解，**需要前端支持`LaTex`的数学公式，并且渲染好看的样式**。

### 1.2 效果展示

数学公式分为行内公式和跨行公式，当然都需要支持和渲染。

我准备了 3 条公式，分别是行内公式、跨行公式和超长的跨行公式：

```
$\alpha+\beta=\gamma$

$$\alpha+\beta=\gamma$$

$$\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}\int_{0}^{1}f(x)dx \sum_{1}^{2}$$
```

这篇测试文章的地址是:[`https://godbmw.com/passage/12`](https://godbmw.com/passage/12)。效果图如下所示：
![](https://static.godbmw.com/images/网站搭建与运营/MathJax：让前端支持数学公式/1.png)

## 2. 使用 MathJax

### 2.1 引入 CDN

在使用 MathJax 之前，需要通过 CDN 引入, 在`<body>`标签中添加：
`<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>`。

### 2.2 配置 MathJax

将 MathJax 的配置封装成一个函数：

```javascript
let isMathjaxConfig = false; // 防止重复调用Config，造成性能损耗

const initMathjaxConfig = () => {
  if (!window.MathJax) {
    return;
  }
  window.MathJax.Hub.Config({
    showProcessingMessages: false, //关闭js加载过程信息
    messageStyle: "none", //不显示信息
    jax: ["input/TeX", "output/HTML-CSS"],
    tex2jax: {
      inlineMath: [["$", "$"], ["\\(", "\\)"]], //行内公式选择符
      displayMath: [["$$", "$$"], ["\\[", "\\]"]], //段内公式选择符
      skipTags: ["script", "noscript", "style", "textarea", "pre", "code", "a"] //避开某些标签
    },
    "HTML-CSS": {
      availableFonts: ["STIX", "TeX"], //可选字体
      showMathMenu: false //关闭右击菜单显示
    }
  });
  isMathjaxConfig = true; //
};
```

### 2.3 使用 MathJax 渲染

MathJax 提供了`window.MathJax.Hub.Queue`来执行渲染。在执行完文本获取操作后，进行渲染操作：

```javascript
if (isMathjaxConfig === false) {
  // 如果：没有配置MathJax
  initMathjaxConfig();
}

// 如果，不传入第三个参数，则渲染整个document
// 因为使用的Vuejs，所以指明#app，以提高速度
window.MathJax.Hub.Queue([
  "Typeset",
  MathJax.Hub,
  document.getElementById("app")
]);
```

### 2.4 修改默认样式

`MathJax`默认样式在被鼠标`focus`的时候，会有蓝色边框出现。对于超长的数学公式，x 方向也会溢出。

添加以下样式代码，覆盖原有样式，从而解决上述问题：

```css
/* MathJax v2.7.5 from 'cdnjs.cloudflare.com' */
.mjx-chtml {
  outline: 0;
}
.MJXc-display {
  overflow-x: auto;
  overflow-y: hidden;
}
```

## 3. 注意事项

### 3.1 不要使用`npm`

**不要使用 npm，会有报错，google 了一圈也没找到解决方案，github 上源码地址有对应的`issue`还没解决**。

博主多次尝试也没有找到解决方法，坐等版本更新和大神指点。

### 3.2 动态数据

在 SPA 单页应用中，数据是通过`Ajax`获取的。此时，**需要在数据获取后，再执行渲染**。

如果习惯`es5`，可以在回调函数中调用`window.MathJax.Hub.Queue`。但是更推荐`es6`，配合`Promise`和`async/await`来避免“回调地域”。

### 3.3 版本问题

对于不同版本或者不同 CDN 的`MathJax`，第二部分的样式覆盖的`class`名称不同。比如在`cdnboot`的`v2.7.0`版本中，样式覆盖的代码应该是下面这段：

```css
/* MathJax v2.7.0 from 'cdn.bootcss.com' */
.MathJax {
  outline: 0;
}
.MathJax_Display {
  overflow-x: auto;
  overflow-y: hidden;
}
```

## 4. 更多资料

- [前端整合 MathjaxJS 的配置笔记](https://www.linpx.com/p/front-end-integration-mathjaxjs-configuration.html)
- [Mathjax 官网](https://www.mathjax.org/)
- [Mathjax 中文文档](https://mathjax-chinese-doc.readthedocs.io/en/latest/)
