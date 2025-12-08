---
title: "十五：开发模式与webpack-dev-server"
date: 2018-10-19
permalink: /2018-10-19-webpack-dev-server/
categories: ["A开源技术课程", "webpack4系列教程"]
---

> 为什么需要开发模式？

借助`webpack`，在开发模式下我们可以使用热重载、路由重定向、代理服务器等功能，而`source-map`更是准确定位代码错误的利器。

<!-- more -->

## 0. 课程介绍和资料

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo15)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课的代码目录如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/38.png)

本节课用的 plugin 和 loader 的配置文件`package.json`如下：

```json
{
  "scripts": {
    "dev": "webpack-dev-server --open"
  },
  "devDependencies": {
    "clean-webpack-plugin": "^0.1.19",
    "html-webpack-plugin": "^3.2.0",
    "jquery": "^3.3.1",
    "webpack": "^4.16.1",
    "webpack-cli": "^3.1.0",
    "webpack-dev-server": "^3.1.4"
  }
}
```

## 1. 为什么需要开发模式？

在之前的课程中，我们都没有指定参数`mode`。但是执行`webpack`进行打包的时候，自动设置为`production`，但是控制台会爆出`warning`的提示。**而开发模式就是指定`mode`为`development`。**

在开发模式下，我们需要对代码进行调试。对应的配置就是：`devtool`设置为`source-map`。在非开发模式下，需要关闭此选项，以减小打包体积。

在开发模式下，还需要热重载、路由重定向、挂代理等功能，`webpack4`已经提供了`devServer`选项，启动一个本地服务器，让开发者使用这些功能。

## 2. 如何使用开发模式？

根据文章开头的`package.json`的配置，只需要在命令行输入：`npm run dev`即可启动开发者模式。

启动效果如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/39.png)

虽然控制台输出了打包信息（假设我们已经配置了热重载），但是磁盘上并没有创建`/dist/`文件夹和打包文件。**控制台的打包文件的相关内容是存储在内存之中的。**

## 3. 编写一些需要的文件

首先，编写一下入口的 html 文件：

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <title>Document</title>
  </head>
  <body>
    This is Index html
  </body>
</html>
```

然后，按照项目目录，简单封装下`/vendor/`下的三个 js 文件，以方便`app.js`调用：

```javascript
// minus.js
module.exports = function(a, b) {
  return a - b;
};

// multi.js
define(function(require, factory) {
  "use strict";
  return function(a, b) {
    return a * b;
  };
});

// sum.js
export default function(a, b) {
  console.log("I am sum.js");
  return a + b;
}
```

好了，准备进入正题。

## 4. 编写 webpack 配置文件

### 4.1 配置代码

_由于配置内容有点多，所以放代码，再放讲解。_

`webpack.config.js`配置如下所示：

```javascript
const webpack = require("webpack");
const HtmlWebpackPlugin = require("html-webpack-plugin");

const path = require("path");

module.exports = {
  entry: {
    app: "./app.js"
  },
  output: {
    publicPath: "/",
    path: path.resolve(__dirname, "dist"),
    filename: "[name]-[hash:5].bundle.js",
    chunkFilename: "[name]-[hash:5].chunk.js"
  },
  mode: "development", // 开发模式
  devtool: "source-map", // 开启调试
  devServer: {
    contentBase: path.join(__dirname, "dist"),
    port: 8000, // 本地服务器端口号
    hot: true, // 热重载
    overlay: true, // 如果代码出错，会在浏览器页面弹出“浮动层”。类似于 vue-cli 等脚手架
    proxy: {
      // 跨域代理转发
      "/comments": {
        target: "https://m.weibo.cn",
        changeOrigin: true,
        logLevel: "debug",
        headers: {
          Cookie: ""
        }
      }
    },
    historyApiFallback: {
      // HTML5 history模式
      rewrites: [{ from: /.*/, to: "/index.html" }]
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: "index.html",
      template: "./index.html",
      chunks: ["app"]
    }),
    new webpack.HotModuleReplacementPlugin(),
    new webpack.NamedModulesPlugin(),
    new webpack.ProvidePlugin({
      $: "jquery"
    })
  ]
};
```

### 4.2 模块热更新

模块热更新需要`HotModuleReplacementPlugin`和`NamedModulesPlugin`这两个插件，并且顺序不能错。并且指定`devServer.hot`为`true`。

有了这两个插件，在项目的 js 代码中可以针对侦测到变更的文件并且做出相关处理。

比如，我们启动开发模式后，修改了`vendor/sum.js`这个文件，此时，需要在浏览器的控制台打印一些信息。那么，`app.js`中就可以这么写：

```javascript
if (module.hot) {
  // 检测是否有模块热更新
  module.hot.accept("./vendor/sum.js", function() {
    // 针对被更新的模块, 进行进一步操作
    console.log("/vendor/sum.js is changed");
  });
}
```

每当`sum.js`被修改后，都可以自动执行回调函数。

### 4.3 跨域代理

随着前后端分离开发的普及，跨域请求变得越来越常见。为了快速开发，可以利用`devServer.proxy`做一个代理转发，来绕过浏览器的跨域限制。

按照前面的配置文件，如果想调用微博的一个接口：`https://m.weibo.cn/comments/hotflow`。只需要在代码中对`/comments/hotflow`进行请求即可：

```javascript
$.get(
  "/comments/hotflow",
  {
    id: "4263554020904293",
    mid: "4263554020904293",
    max_id_type: "0"
  },
  function(data) {
    console.log(data);
  }
);
```

### 4.4 HTML5--History

当项目使用`HTML5 History API` 时，任意的 404 响应都可能需要被替代为 `index.html`。

在 SPA（单页应用）中，任何响应直接被替代为`index.html`。

在 vuejs 官方的脚手架`vue-cli`中，开发模式下配置如下：

```javascript
// ...
historyApiFallback: {
  rewrites: [{ from: /.*/, to: "/index.html" }];
}
// ...
```

## 5. 编写入口文件

最后，在前面所有的基础上，让我们来编写下入口文件`app.js`：

```javascript
import sum from "./vendor/sum";
console.log("sum(1, 2) = ", sum(1, 2));
var minus = require("./vendor/minus");
console.log("minus(1, 2) = ", minus(1, 2));
require(["./vendor/multi"], function(multi) {
  console.log("multi(1, 2) = ", multi(1, 2));
});

$.get(
  "/comments/hotflow",
  {
    id: "4263554020904293",
    mid: "4263554020904293",
    max_id_type: "0"
  },
  function(data) {
    console.log(data);
  }
);

if (module.hot) {
  // 检测是否有模块热更新
  module.hot.accept("./vendor/sum.js", function() {
    // 针对被更新的模块, 进行进一步操作
    console.log("/vendor/sum.js is changed");
  });
}
```

## 6. 效果检测

在命令行键入：`npm run dev`开启服务器后，会自动打开浏览器。如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/40.png)

打开控制台，可以看到代码都正常运行没有出错。除此之外，由于开启了`source-map`，所以可以定位代码位置（下图绿框内）：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/41.png)

## 7. 参考资料

- dev-server 文档: [https://www.webpackjs.com/configuration/dev-server/](https://www.webpackjs.com/configuration/dev-server/)
- 开发模式 文档:[https://www.webpackjs.com/guides/development/](https://www.webpackjs.com/guides/development/)
