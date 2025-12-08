---
title: "CSS3盒模型：border-box"
date: 2018-06-05
permalink: /2018-06-05-border-sizing/
---
`box-sizing`可以声明计算元素高宽的 CSS 盒模型。它有`content-box`、`border-box`和`inherit`三种取值。其中`border-box`是 css3 新增，也是主流 UI 框架的全局默认属性。


## 两种盒模型


### `content-box`


默认值，也是 css2.1 中的盒子模型。在计算`width`和`height`时候，不计算`border`、`padding`和`margin`。高度、宽度都只是内容高度。


### `border-box`


`css3`新增。 `width`和`height`属性包括内容，内边距和边框，但不包括外边距。


它的计算公式是：

1. width = border + padding + 内容宽度
2. height = border + padding + 内容高度

## 为什么不计算`margin`


从上面可以知道，即时是`border-box`也是不计算`margin`，只是多余计算了`border`和`padding`。因为`border`和`padding`都是盒子模型的一部分，但是`margin`标记的是盒子和盒子的间距。所以，**`border-box`****的计算方法更符合****`box-sizing`****的语义**。


> 问题来了，如果有时候一定要设置margin，怎么做到自由控制来保证兼容？例如，我们下面要设置一个撑满页面的盒子元素，而且有外边距干扰，怎么做？


## 实际应用


根据项目中的使用经验和 w3c 的建议，推荐`box-sizing`属性设置为`border-box`。在样式表文件中添加以下代码：


```css
* {
    margin: 0;
    padding: 0;
}
div {
    box-sizing: border-box;
}
```


除了通用代码，`border-box`还可以配合 css3 中的四则运算符`calc`来使用，来实现对 margin 的控制。


代码如下：


```html
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <meta http-equiv="X-UA-Compatible" content="ie=edge" />
        <title>dongyuanxin.github.io</title>
        <style type="text/css">
            * {
                margin: 0;
                padding: 0;
            }
            #app {
                box-sizing: border-box; /* 指定计算方式 */
                margin: 10px; /* 外边距 */
                /* 利用 css3 的 calc */
                width: calc(100vw - 2 * 10px);
                height: calc(100vh - 2 * 10px);
            }
        </style>
    </head>
    <body>
        <div id="app"></div>
    </body>
</html>

```


