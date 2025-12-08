---
title: "十二：处理第三方JavaScript库"
date: 2018-10-09
permalink: /2018-10-09-webpack-js-pacakge/
categories: ["A开源技术课程", "webpack4系列教程"]
---

项目做大之后，开发者会更多专注在业务逻辑上，其他方面则尽力使用第三方`JS`库来实现。

由于`js`变化实在太快，所以出现了多种引入和管理第三方库的方法，常用的有 3 中：

1. CDN：`<script></script>`标签引入即可
2. npm 包管理： 目前最常用和最推荐的方法
3. 本地`js`文件：一些库由于历史原因，没有提供`es6`版本，需要手动下载，放入项目目录中，再手动引入。

> 本文详细介绍了：在上面 3 种方法的基础上，如何配合`webpack`更优雅地引入和使用第三方`js`库。

<!-- more -->

## 0. 课程介绍和资料

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo12)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课的代码目录如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/27.png)

本节课的`package.json`内容如下：

```json
{
  "dependencies": {
    "jquery": "^3.3.1"
  },
  "devDependencies": {
    "webpack": "^4.16.1"
  }
}
```

## 1. 如何使用和管理第三方`JS`库？

项目做大之后，开发者会更多专注在业务逻辑上，其他方面则尽力使用第三方`JS`库来实现。

由于`js`变化实在太快，所以出现了多种引入和管理第三方库的方法，常用的有 3 中：

1. CDN：`<script></script>`标签引入即可
2. npm 包管理： 目前最常用和最推荐的方法
3. 本地`js`文件：一些库由于历史原因，没有提供`es6`版本，需要手动下载，放入项目目录中，再手动引入。

针对第一种和第二种方法，各有优劣，有兴趣可以看这篇：[《CDN 使用心得：加速双刃剑》](https://godbmw.com/passage/60)

**针对第三种方法，如果没有`webpack`，则需要手动引入`import`或者`require`来加载文件；但是，`webpack`提供了`alias`的配置，配合`webpack.ProvidePlugin`这款插件，可以跳过手动入，直接使用！**

## 2. 编写入口文件

如项目目录图片所展示的，我们下载了`jquery.min.js`，放到了项目中。同时，我们也通过`npm`安装了`jquery`。

**为了尽可能模仿生产环境，`app.js`中使用了`$`来调用 jq，还使用了`jQuery`来调用 jq。**

因为正式项目中，由于需要的依赖过多，挂载到`window`对象的库，很容易发生命名冲突问题。此时，就需要重命名库。例如：`$`就被换成了`jQuery`。

```javascript
// app.js
$("div").addClass("new");

jQuery("div").addClass("old");

// 运行webpack后
// 浏览器打开 index.html, 查看 div 标签的 class
```

## 3. 编写配置文件

`webpack.ProvidePlugin`参数是键值对形式，键就是我们项目中使用的变量名，值就是键所指向的库。

`webpack.ProvidePlugin`会先从`npm`安装的包中查找是否有符合的库。

如果`webpack`配置了`resolve.alias`选项（理解成“别名”），那么`webpack.ProvidePlugin`就会顺着这条链一直找下去。

```javascript
// webpack.config.js
const path = require("path");
const webpack = require("webpack");

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
  resolve: {
    alias: {
      jQuery$: path.resolve(__dirname, "src/vendor/jquery.min.js")
    }
  },
  plugins: [
    new webpack.ProvidePlugin({
      $: "jquery", // npm
      jQuery: "jQuery" // 本地Js文件
    })
  ]
};
```

## 4. 结果分析和验证

老规矩，根绝上面配置，先编写一下`index.html`：

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
    <div></div>
    <script src="./dist/app.bundle.js"></script>
  </body>
</html>
```

命令行运行`webpack`进行项目打包：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/28.png)

在 Chrome 中打开`index.html`。如下图所示，`<div>`标签已经被添加上了`old`和`new`两个样式类。证明在`app.js`中使用的`$`和`jQuery`都成功指向了`jquery`库。

![](https://static.godbmw.com/images/webpack/webpack4系列教程/29.png)
