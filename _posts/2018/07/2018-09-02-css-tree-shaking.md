---
title: "九：CSS-Tree-Shaking"
date: 2018-09-02
permalink: /2018-09-02-css-tree-shaking/
categories: ["A开源技术课程", "webpack4系列教程"]
---

> CSS 也有 Tree Shaking？

是滴，随着 webpack 的兴起，css 也可以进行 Tree Shaking： 以去除项目代码中用不到的 CSS 样式，仅保留被使用的样式代码。

通常来说，造成学习和理解难度的原因，无非是无法模拟较真的生产环境来进行演练 （比如：在 js、html 等文件中使用 css 样式）。

但是，本篇博文已经帮您准备好了。快来看看吧！

<!-- more -->

### 0. 课程介绍和资料

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo09)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本次课程的代码目录(如下图所示)：
![](https://static.godbmw.com/images/webpack/webpack4系列教程/18.png)

### 1. CSS 也有 Tree Shaking？

> 是滴，随着 webpack 的兴起，css 也可以进行 Tree Shaking： 以去除项目代码中用不到的 CSS 样式，仅保留被使用的样式代码。

为了方便理解 Tree Shaking 概念，并且与 JS Tree Shaking 进行横向比较，请查看：[webpack4 系列教程\(八\): JS Tree Shaking](https://godbmw.com/passage/48)

### 2. 项目环境仿真

因为 CSS Tree Shaking 并不像 JS Tree Shaking 那样方便理解，所以首先要先模拟一个真实的项目环境，来体现 CSS 的 Tree Shaking 的配置和效果。

我们首先编写 `/src/css/base.css` 样式文件，在文件中，我们编写了 3 个样式类。但在代码中，我们只会使用 `.box` 和 `.box--big` 这两个类。代码如下所示：

```css
/* base.css */
html {
  background: red;
}

.box {
  height: 200px;
  width: 200px;
  border-radius: 3px;
  background: green;
}

.box--big {
  height: 300px;
  width: 300px;
  border-radius: 5px;
  background: red;
}

.box-small {
  height: 100px;
  width: 100px;
  border-radius: 2px;
  background: yellow;
}
```

按照正常使用习惯，DOM 操作来实现样式的添加和卸载，是一贯技术手段。所以，入口文件 `/src/app.js` 中创建了一个 `<div>` 标签，并且将它的类设为 `.box`

```javascript
// app.js

import base from "./css/base.css";

var app = document.getElementById("app");
var div = document.createElement("div");
div.className = "box";
app.appendChild(div);
```

最后，为了让环境更接近实际环境，我们在`index.html`的一个标签，也引用了定义好的 `box-big` 样式类。

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <link rel="stylesheet" href="./dist/app.min.css" />
    <title>Document</title>
  </head>
  <body>
    <div id="app">
      <div class="box-big"></div>
    </div>
    <script src="./dist/app.bundle.js"></script>
  </body>
</html>
```

按照我们的仿真的环境，最终 Tree Shaking 之后的效果应该是：**打包后的 css 文件不含有 `box-small` 样式类**。下面，就实现这个效果！

### 3. 认识下 `PurifyCSS`

没错，就是这货帮助我们进行 CSS Tree Shaking 操作。为了能准确指明要进行 Tree Shaking 的 CSS 文件，它还有好朋友 `glob-all` （另一个第三方库）。

`glob-all` 的作用就是帮助 `PurifyCSS` 进行路径处理，定位要做 Tree Shaking 的路径文件。

它们俩搭配起来，画风如下：

```javascript
const PurifyCSS = require("purifycss-webpack");
const glob = require("glob-all");

let purifyCSS = new PurifyCSS({
  paths: glob.sync([
    // 要做CSS Tree Shaking的路径文件
    path.resolve(__dirname, "./*.html"),
    path.resolve(__dirname, "./src/*.js")
  ])
});
```

好了，这只是一个小小的 demo。下面我们要把它用到我们的`webpack.config.js`中来。

### 4. 编写配置文件

为了方便最后检查打包后的 css 文件，配置中还使用了 `extract-text-webpack-plugin` 这个插件。如果忘记了它的用法，请查看：

- [webpack4 系列教程\(六\): 处理 SCSS](https://godbmw.com/passage/37)
- [webpack4 系列教程\(五\): 处理 CSS](https://godbmw.com/passage/36)

所以，我们的`package.json`文件如下：

```json
{
  "devDependencies": {
    "css-loader": "^1.0.0",
    "extract-text-webpack-plugin": "^4.0.0-beta.0",
    "glob-all": "^3.1.0",
    "purify-css": "^1.2.5",
    "purifycss-webpack": "^0.7.0",
    "style-loader": "^0.21.0",
    "webpack": "^4.16.0"
  }
}
```

安装完相关插件后，我们需要在 webpack 的`plugins`配置中引用第三部分定义的代码。

然后结合`extract-text-webpack-plugin`的配置，编写如下`webpack.config.js`:

```javascript
// webpack.config.js
const path = require("path");
const PurifyCSS = require("purifycss-webpack");
const glob = require("glob-all");
const ExtractTextPlugin = require("extract-text-webpack-plugin");

let extractTextPlugin = new ExtractTextPlugin({
  filename: "[name].min.css",
  allChunks: false
});

let purifyCSS = new PurifyCSS({
  paths: glob.sync([
    // 要做CSS Tree Shaking的路径文件
    path.resolve(__dirname, "./*.html"), // 请注意，我们同样需要对 html 文件进行 tree shaking
    path.resolve(__dirname, "./src/*.js")
  ])
});

module.exports = {
  entry: {
    app: "./src/app.js"
  },
  output: {
    publicPath: __dirname + "/dist/",
    path: path.resolve(__dirname, "dist"),
    filename: "[name].bundle.js",
    chunkFilename: "[name].chunk.js"
  },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ExtractTextPlugin.extract({
          fallback: {
            loader: "style-loader",
            options: {
              singleton: true
            }
          },
          use: {
            loader: "css-loader",
            options: {
              minimize: true
            }
          }
        })
      }
    ]
  },
  plugins: [extractTextPlugin, purifyCSS]
};
```

### 5. 结果分析

命令行运行`webpack`打包后，样式文件被抽离到了 `/dist/app.min.css` 文件中。文件内容如下图所示（_肯定好多朋友懒得手动打包_）：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/17.png)

我们在`index.html` 和 `src/app.js` 中引用的样式都被打包了，而没有被使用的样式类--`box-small`，就没有出现在图片中。成功！

_终_
