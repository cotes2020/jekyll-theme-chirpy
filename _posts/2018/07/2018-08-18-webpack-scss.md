---
title: "六：处理SCSS"
date: 2018-08-18
permalink: /2018-08-18-webpack-scss/
categories: ["开源技术课程", "webpack4系列教程"]
---

这节课以 SCSS 为例，讲解如何在`webpack`中编译这种 CSS 预处理语言，并配合`CSS`的 LOADER 来进行组合处理。一些更复杂的应用，请翻看《`webpack`处理 CSS》这篇文章。

[>>> 了解更多处理`css`的内容](https://dongyuanxin.github.io/2018-08-17-webpack-css/)

<!-- more -->

> 这节课讲解`webpack4`中处理`scss`。只需要在处理`css`的配置上增加编译`scss`的 LOADER 即可。[了解更多处理`css`的内容 >>>](https://dongyuanxin.github.io/2018-08-17-webpack-css/)

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo06)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

### 1. 准备工作

为了方便叙述，这次代码目录的样式文件只有一个`scss`文件，以帮助我们了解核心 LOADER 的使用。

下图展示了这次的目录代码结构：
![](https://static.godbmw.com/images/webpack/webpack4系列教程/10.png)

这次我们需要用到`node-sass`，`sass-loader`等 LOADER，`package.json`如下：

```json
{
  "devDependencies": {
    "css-loader": "^1.0.0",
    "extract-text-webpack-plugin": "^4.0.0-beta.0",
    "node-sass": "^4.9.2",
    "sass-loader": "^7.0.3",
    "style-loader": "^0.21.0",
    "webpack": "^4.16.0"
  }
}
```

其中，`base.scss`代码如下：

```scss
$bgColor: red !default;
*,
body {
  margin: 0;
  padding: 0;
}
html {
  background-color: $bgColor;
}
```

`index.html`代码如下：

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <title>Document</title>
  </head>
  <body>
    <script src="./dist/app.bundle.js"></script>
  </body>
</html>
```

### 2. 编译打包`scss`

首先，在入口文件`app.js`中引入我们的 scss 样式文件：

```javascript
import "./scss/base.scss";
```

下面，开始编写`webpack.config.js`文件:

```javascript
const path = require("path");

module.exports = {
  entry: {
    app: "./src/app.js"
  },
  output: {
    publicPath: __dirname + "/dist/",
    path: path.resolve(__dirname, "dist"),
    filename: "[name].bundle.js"
  },
  module: {
    rules: [
      {
        test: /\.scss$/,
        use: [
          {
            loader: "style-loader" // 将 JS 字符串生成为 style 节点
          },
          {
            loader: "css-loader" // 将 CSS 转化成 CommonJS 模块
          },
          {
            loader: "sass-loader" // 将 Sass 编译成 CSS
          }
        ]
      }
    ]
  }
};
```

需要注意的是，`module.rules.use`数组中，loader 的位置。根据 webpack 规则：**放在最后的 loader 首先被执行**。所以，首先应该利用`sass-loader`将 scss 编译为 css，剩下的配置和处理 css 文件相同。

### 3. 检查打包结果

因为 scss 是 css 预处理语言，所以我们要检查下打包后的结果，打开控制台，如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/9.png)

同时，对于其他的 css 预处理语言，处理方式一样，首先应该编译成 css，然后交给 css 的相关 loader 进行处理。[点我了解更多关于处理`css`的内容 >>>](http://dongyuanxin.github.io/#/passage/36)
