---
title: "一：打包JS"
date: 2018-07-30
permalink: /2018-07-30-webpack-pack-js/
categories: ["开源技术课程", "webpack4系列教程"]
---

> webpack 本身就是为了打包`js`所设计，作为第一节，介绍**怎么打包`js`**。

### 1. 检验`webpack`规范支持

`webpack`支持`es6`, `CommonJS`, `AMD`。

创建`vendor`文件夹，其中`minus.js`、`multi.js`和`sum.js`分别用 CommonJS、AMD 和 ES6 规范编写。

[>>> vendor 文件夹 代码地址](https://github.com/dongyuanxin/webpack-demos/tree/master/demo01/vendor)

在入口文件`app.js`中，我们分别用 3 中规范，引用`vendor`文件夹中的 js 文件。

```javascript
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

### 2. 编写配置文件

`webpack.config.js`是 webpack 默认的配置文件名，[>>> webpack.config.js 代码地址](https://github.com/dongyuanxin/webpack-demos/blob/master/demo01/webpack.config.js)，其中配置如下：

```javascript
const path = require("path");

module.exports = {
  entry: {
    app: "./app.js"
  },
  output: {
    publicPath: __dirname + "/dist/", // js引用路径或者CDN地址
    path: path.resolve(__dirname, "dist"), // 打包文件的输出目录
    filename: "bundle.js"
  }
};
```

注意`output.publicPath`参数，代表：**`js`文件内部引用其他文件的路径**。

### 3. 收尾

打包后的`js`文件会按照我们的配置放在`dist`目录下，这时，**需要创建一个`html`文件，引用打包好的`js`文件**。

然后在 Chrome 打开(**这节课只是打包`js`,不包括编译`es6`**)，就可以看到我们代码的运行结果了。

### 4. 更多

本节的代码地址：[>>> 点我进入](https://github.com/dongyuanxin/webpack-demos/tree/master/demo01)

项目的代码仓库：[>>> 点我进入](https://github.com/dongyuanxin/webpack-demos)
