---
title: "七：SCSS提取和懒加载"
date: 2018-08-28
permalink: /2018-08-28-webpack-scss-lazy/
categories: ["A开源技术课程", "webpack4系列教程"]
---

本节课讲解在`webpack v4`中的 SCSS 提取和懒加载。值得一提的是，`v4`和`v3`在 Scss 的懒加载上的处理方法有着巨大差别：

- 入口文件需要引用相关 LOADER 的 css 文件
- 配置需要安装针对`v4`版本的`extract-text-webpack-plugin`

<!-- more -->

> 本节课讲解在`webpack v4`中的 SCSS 提取和懒加载。值得一提的是，`v4`和`v3`在 Scss 的懒加载上的处理方法有着巨大差别。

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo07)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

### 1. 准备工作

关于 SCSS 处理的基础，请参考[webpack4 系列教程(六): 处理 SCSS](https://godbmw.com/passage/37)。

本节课主要涉及 SCSS 在懒加载下提取的相关配置和插件使用。

下图展示了这次的目录代码结构：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/11.png)

为了实现 SCSS 懒加载，我们使用了`extract-text-webpack-plugin`插件。

需要注意，**在安装插件的时候，应该安装针对`v4`版本的`extract-text-webpack-plugin`**。npm 运行如下命令：`npm install --save-dev extract-text-webpack-plugin@next`

其余配置，与第六课相似。`package.json`配置如下：

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

关于我们的 scss 文件下的样式文件，`base.scss`:

```scss
// 网站默认背景色：red
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

`common.scss`:

```scss
// 覆盖原来颜色：green
html {
  background-color: green !important;
}
```

### 2. 使用`ExtractTextPlugin`

使用`extract-text-webpack-plugin`，需要在`webpack.config.js`的`plugins`选项和`rules`中`scss`的相关选项进行配置。

`webpack.config.js`:

```javascript
const path = require("path");
const ExtractTextPlugin = require("extract-text-webpack-plugin");

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
        test: /\.scss$/,
        use: ExtractTextPlugin.extract({
          // 注意 1
          fallback: {
            loader: "style-loader"
          },
          use: [
            {
              loader: "css-loader",
              options: {
                minimize: true
              }
            },
            {
              loader: "sass-loader"
            }
          ]
        })
      }
    ]
  },
  plugins: [
    new ExtractTextPlugin({
      filename: "[name].min.css",
      allChunks: false // 注意 2
    })
  ]
};
```

在配置中，**注意 1**中的`callback`配置项，针对 不提取为单独`css`文件的`scss`样式 应该使用的 LOADER。即使用`style-loader`将 scss 处理成 css 嵌入网页代码。

**注意 2**中的`allChunks`必须指明为`false`。否则会包括异步加载的 CSS！

### 3. `SCSS`引用和懒加载

在项目入口文件`app.js`中，针对 scss 懒加载，需要引入以下配置代码：

```javascript
import "style-loader/lib/addStyles";
import "css-loader/lib/css-base";
```

剩下我们先设置背景色为红色，在用户点击鼠标后，懒加载`common.scss`，使背景色变为绿色。剩余代码如下：

```javascript
import "./scss/base.scss";

var loaded = false;
window.addEventListener("click", function() {
  if (!loaded) {
    import(/* webpackChunkName: 'style'*/ "./scss/common.scss").then(_ => {
      // chunk-name : style
      console.log("Change bg-color of html");
      loaded = true;
    });
  }
});
```

### 4. 打包和引入

根据我们在`app.js`中的`webpackChunkName`的配置，可以猜测，打包结果中有：`style.chunk.js` 文件。

命令行执行`webpack`打包后，`/dist/`目录中的打包结果如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/12.png)

最后，我们需要在 html 代码中引入打包结果中的、**非懒加载**的样式(`/dist/app.min.css`)和 js 文件(`/dist/app.bundle.js`)。

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <title>Document</title>
    <link rel="stylesheet" href="./dist/app.min.css" />
  </head>
  <body>
    <script src="./dist/app.bundle.js"></script>
  </body>
</html>
```

_终_
