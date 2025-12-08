---
title: "十四：Clean Plugin and Watch Mode"
date: 2018-10-18
permalink: /2018-10-18-webpack-clean-and-watch-mode/
categories: ["A开源技术课程", "webpack4系列教程"]
---

> 简单来说：生产开发过程中优雅地自动化！！！

在实际开发中，由于需求变化，会经常改动代码，然后用 webpack 进行打包发布。由于改动过多，我们`/dist/`目录中会有很多版本的代码堆积在一起，乱七八糟。

为了让打包目录更简洁，需要`Clean Plugin`，在每次打包前，自动清理`/dist/`目录下的文件。

除此之外，借助 webpack 命令本身的命令参数--`Watch Mode`：监察你的所有文件,任一文件有所变动,立刻重新自动打包。

<!-- more -->

## 0. 课程介绍和资料

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo14)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课的代码目录如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/34.png)

本节课用的 plugin 和 loader 的配置文件`package.json`如下：

```json
{
  "devDependencies": {
    "clean-webpack-plugin": "^0.1.19",
    "html-webpack-plugin": "^3.2.0",
    "webpack": "^4.16.1"
  }
}
```

## 1. 什么是`Clean Plugin`和`Watch Mode`？

在实际开发中，由于需求变化，会经常改动代码，然后用 webpack 进行打包发布。由于改动过多，我们`/dist/`目录中会有很多版本的代码堆积在一起，乱七八糟。

为了让打包目录更简洁，**这时候需要`Clean Plugin`，在每次打包前，自动清理`/dist/`目录下的文件。**

除此之外，借助 webpack 命令本身的命令参数，**可以开启`Watch Mode`：监察你的所有文件,任一文件有所变动,它就会立刻重新自动打包。**

## 2. 编写入口文件和 js 脚本

入口文件`app.js`代码：

```javascript
console.log("This is entry js");

// ES6
import sum from "./vendor/sum";
console.log("sum(1, 2) = ", sum(1, 2));

// CommonJs
var minus = require("./vendor/minus");
console.log("minus(1, 2) = ", minus(1, 2));

// AMD
require(["./vendor/multi"], function(multi) {
  console.log("multi(1, 2) = ", multi(1, 2));
});
```

`vendor/sum.js`:

```javascript
export default function(a, b) {
  return a + b;
}
```

`vendor/multi.js`:

```javascript
define(function(require, factory) {
  "use strict";
  return function(a, b) {
    return a * b;
  };
});
```

`vendor/minus.js`:

```javascript
module.exports = function(a, b) {
  return a - b;
};
```

## 3. 编写 webpack 配置文件

`CleanWebpackPlugin`参数传入数组，其中每个元素是每次需要清空的文件目录。

需要注意的是：**应该把`CleanWebpackPlugin`放在`plugin`配置项的最后一个**，因为 webpack 配置是倒序的（最后配置的最先执行）。以保证每次正式打包前，先清空原来遗留的打包文件。

```javascript
const webpack = require("webpack");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const CleanWebpackPlugin = require("clean-webpack-plugin");

const path = require("path");

module.exports = {
  entry: {
    app: "./app.js"
  },
  output: {
    publicPath: __dirname + "/dist/", // js引用路径或者CDN地址
    path: path.resolve(__dirname, "dist"), // 打包文件的输出目录
    filename: "[name]-[hash:5].bundle.js",
    chunkFilename: "[name]-[hash:5].chunk.js"
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: "index.html",
      template: "./index.html",
      chunks: ["app"]
    }),
    new CleanWebpackPlugin(["dist"])
  ]
};
```

执行`webpack`打包，在控制台会首先输出一段关于相关文件夹已经清空的的提示，如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/35.png)

## 4. 开启`Watch Mode`

直接在`webpack`命令后加上`--watch`参数即可：`webpack --watch`。

控制台会提示用户“开启 watch”。我改动了一次文件，改动被 webpack 侦听到，就会自动重新打包。如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/36.png)

如果想看到详细的打包过程，可以使用：`webpack -w --progress --display-reasons --color`。控制台就会以花花绿绿的形式展示出打包过程，看起来比较酷炫：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/37.png)
