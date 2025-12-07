---
title: "四：单页面解决方案--代码分割和懒加载"
date: 2018-08-08
permalink: /2018-08-08-webpack-spa-split-lazy/
categories: ["开源技术课程", "webpack4系列教程"]
---

> 本节课讲解`webpack4`打包**单页应用**过程中的代码分割和代码懒加载。不同于多页面应用的提取公共代码，单页面的代码分割和懒加载不是通过`webpack`配置来实现的，而是通过`webpack`的写法和内置函数实现的。

目前`webpack`针对此项功能提供 2 种函数：

1.  `import()`: 引入并且自动执行相关 js 代码
2.  `require.ensure()`: 引入但需要手动执行相关 js 代码

本文将会进行逐一讲解。

<!-- more -->

> 本节课讲解`webpack4`打包**单页应用**过程中的代码分割和代码懒加载。不同于多页面应用的提取公共代码，单页面的代码分割和懒加载不是通过`webpack`配置来实现的，而是通过`webpack`的写法和内置函数实现的。

目前`webpack`针对此项功能提供 2 种函数：

1.  `import()`: 引入并且自动执行相关 js 代码
2.  `require.ensure()`: 引入但需要手动执行相关 js 代码

本文将会进行逐一讲解。

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo04)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

### 1. 准备工作

此次代码的目录结构如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/3.png)

其中，`page.js`是入口文件,`subPageA.js`和`subPageB.js`共同引用`module.js`。下面，我们按照代码引用的逻辑，从底向上展示代码：

`module.js`:

```javascript
export default "module";
```

`subPageA.js`:

```javascript
import "./module";
console.log("I'm subPageA");
export default "subPageA";
```

`subPageB.js`:

```javascript
import "./module";
console.log("I'm subPageB");
export default "subPageB";
```

请注意：subPageA.js 和 subPageB.js 两个文件中都执行了`console.log()`语句。之后将会看到`import()`和`require()`不同的表现形式：是否会自动执行 js 的代码？

### 2. 编写配置文件

下面编写`webpack`配置文件（很简单）：

```javascript
const webpack = require("webpack");
const path = require("path");

module.exports = {
  entry: {
    page: "./src/page.js" //
  },
  output: {
    publicPath: __dirname + "/dist/",
    path: path.resolve(__dirname, "dist"),
    filename: "[name].bundle.js",
    chunkFilename: "[name].chunk.js"
  }
};
```

同时，关于第三方库，因为要在`page.js`中使用`lodash`，所以，`package.json`文件配置如下：

```json
{
  "devDependencies": {
    "webpack": "^4.15.1"
  },
  "dependencies": {
    "lodash": "^4.17.10"
  }
}
```

### 3. `import()`编写`page.js`

我个人是非常推荐`import()`写法，因为和 es6 语法看起来很像。除此之外，`import()`可以通过注释的方法来指定打包后的 chunk 的名字。

除此之外，相信对`vue-router`熟悉的朋友应该知道，其官方文档的路由懒加载的配置也是通过`import()`来书写的。

下面，我们将书写`page.js`:

```javascript
import(/* webpackChunkName: 'subPageA'*/ "./subPageA").then(function(subPageA) {
  console.log(subPageA);
});

import(/* webpackChunkName: 'subPageB'*/ "./subPageB").then(function(subPageB) {
  console.log(subPageB);
});

import(/* webpackChunkName: 'lodash'*/ "lodash").then(function(_) {
  console.log(_.join(["1", "2"]));
});
export default "page";
```

命令行中运行`webpack`，打包结果如下：
![](https://static.godbmw.com/images/webpack/webpack4系列教程/4.png)

我们创建`index.html`文件，通过`<script>`标签引入我们打包结果，需要注意的是：因为是单页应用，所以只要引用入口文件即可（即是上图中的`page.bundle.js`）。

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
    <script src="./dist/page.bundle.js"></script>
  </body>
</html>
```

打开浏览器控制台，刷新界面，结果如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/5.png)

图中圈出的部分，就是说明`import()`会自动运行`subPageA.js和subPageB.js`的代码。

在 NetWork 选项中，我们可以看到，懒加载也成功了：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/6.png)

### 4. `require()`编写`page.js`

`require.ensure()`不会自动执行`js`代码，请注意注释：

```javascript
require.ensure(
  ["./subPageA.js", "./subPageB.js"], // js文件或者模块名称
  function() {
    var subPageA = require("./subPageA"); // 引入后需要手动执行，控制台才会打印
    var subPageB = require("./subPageB");
  },
  "subPage" // chunkName
);

require.ensure(
  ["lodash"],
  function() {
    var _ = require("lodash");
    _.join(["1", "2"]);
  },
  "vendor"
);

export default "page";
```

其实，根据我们编写的代码，`subPageA.js`和`subPageB.js`共同引用了`module.js`文件，我们可以将`module.js`体现抽离出来：

```javascript
require.include("./module.js"); // 将subPageA和subPageB共用的module.js打包在此page中

// ...
// 再输入上面那段代码
```

最终打包后，检验和引入方法与`import()`一致，这里不再冗赘。
