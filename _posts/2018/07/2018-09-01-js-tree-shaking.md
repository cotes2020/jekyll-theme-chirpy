---
title: "八：JS Tree Shaking"
date: 2018-09-01
permalink: /2018-09-01-js-tree-shaking/
categories: ["开源技术课程", "webpack4系列教程"]
---

> 本文简述了`webpack3` 和 `webpack4`在 JS Tree Shaking 上的区别，并详细介绍了在 `webpack4` 环境下如何对 JS 代码 和 第三方库 进行 Tree Shaking。

Now, 一起来踩坑吧 ♪(^∇^\*)

<!-- more -->

### 0. 课程介绍和资料

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo08)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本次课程的代码目录(如下图所示)：
![](https://static.godbmw.com/images/webpack/webpack4系列教程/13.png)

### 1. 什么是`Tree Shaking`？

> 字面意思是摇树，一句话：**项目中没有使用的代码会在打包时候丢掉**。JS 的 Tree Shaking 依赖的是 ES2015 的模块系统（比如：`import`和`export`）

本文介绍`Js Tree Shaking`在`webpack v4`中的激活方法。

### 2. 不再需要`UglifyjsWebpackPlugin`

是的，在`webpack v4`中，不再需要配置`UglifyjsWebpackPlugin`。（详情请见：[文档](https://www.webpackjs.com/plugins/uglifyjs-webpack-plugin/)） 取而代之的是，更加方便的配置方法。

只需要配置`mode`为`"production"`，即可显式激活 `UglifyjsWebpackPlugin` 插件。

_注意：根据版本不同，更新的`webpack v4.x`不配置`mode`也会自动激活插件_

我们的`webpack.config.js`配置如下：

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
  mode: "production"
};
```

我们在`util.js`文件中输入以下内容：

```javascript
// util.js
export function a() {
  return 'this is function "a"';
}

export function b() {
  return 'this is function "b"';
}

export function c() {
  return 'this is function "c"';
}
```

然后在`app.js`中引用`util.js`的`function a()`函数：

```javascript
// app.js
import { a } from "./vendor/util";
console.log(a());
```

命令行运行`webpack`打包后，打开打包后生成的`/dist/app.bundle.js`文件。然后，查找我们`a()`函数输出的字符串，如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/14.png)

如果将查找内容换成 `this is function "c"` 或者 `this is function "b"`, 并没有相关查找结果。**说明`Js Tree Shaking`成功**。

### 3. 如何处理第三方`JS`库？

> 对于经常使用的第三方库（例如 jQuery、lodash 等等），如何实现`Tree Shaking`？下面以 lodash.js 为例，进行介绍。

#### 3.1 尝试 `Tree Shaking`

安装 lodash.js : `npm install lodash --save`

在 app.js 中引用 lodash.js 的一个函数：

```javascript
// app.js
import { chunk } from "lodash";
console.log(chunk([1, 2, 3], 2));
```

命令行打包。如下图所示，打包后大小是 70kb。显然，只引用了一个函数，不应该这么大。并没有进行`Tree Shaking`。

![](https://static.godbmw.com/images/webpack/webpack4系列教程/15.png)

#### 3.2 第三方库的模块系统 版本

本文开头讲过，`js tree shaking` 利用的是 es 的模块系统。而 lodash.js 没有使用 CommonJS 或者 ES6 的写法。所以，**安装库对应的模块系统即可**。

安装 lodash.js 的 es 写法的版本：`npm install lodash-es --save`

小小修改一下`app.js`:

```javascript
// app.js
import { chunk } from "lodash-es";
console.log(chunk([1, 2, 3], 2));
```

再次打包，打包结果只有 3.5KB（如下图所示）。显然，`tree shaking`成功。

![](https://static.godbmw.com/images/webpack/webpack4系列教程/16.png)

_友情提示：在一些对加载速度敏感的项目中使用第三方库，请注意库的写法是否符合 es 模板系统规范，以方便`webpack`进行`tree shaking`。_

_终_
