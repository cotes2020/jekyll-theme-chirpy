---
title: "三：多页面解决方案--提取公共代码"
date: 2018-08-06
permalink: /2018-08-06-webpack-mutiple-pages/
categories: ["开源技术课程", "webpack4系列教程"]
---

> 这节课讲解`webpack4`打包多页面应用过程中的**提取公共代码**部分。相比于`webpack3`，`4.0`版本用`optimization.splitChunks`配置替换了`3.0`版本的`CommonsChunkPlugin`插件。在使用和配置上，更加方便和清晰。

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo03)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

代码目录结构如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/2.png)

最终，成功提取公共代码，如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/1.png)

<!-- more -->

> 这节课讲解`webpack4`打包多页面应用过程中的**提取公共代码**部分。相比于`webpack3`，`4.0`版本用`optimization.splitChunks`配置替换了`3.0`版本的`CommonsChunkPlugin`插件。在使用和配置上，更加方便和清晰。

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo03)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

### 1. 准备工作

按照惯例，我们在`src/`文件夹下创建`pageA.js`和`pageB.js`分别作为两个入口文件。同时，这两个入口文件同时引用`subPageA.js`和`subPageB.js`，而`subPageA.js`和`subPageB.js`又同时引用`module.js`文件。

代码目录结构如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/2.png)

希望大家理清逻辑关系，下面从底层往上层进行代码书写。

**`module.js`:**

```javascript
export default "module";
```

**`subPageA.js`:**

```javascript
import "./module";
export default "subPageA";
```

**`subPageB.js`:**

```javascript
import "./module";
export default "subPageB";
```

正如我们所见，`subPageA.js`和`subPageB.js`同时引用`module.js`。

最后，我们封装入口文件。而为了让情况更真实，**这两个入口文件又同时引用了`lodash`这个第三方库**。

**`pageA.js`:**

```javascript
import "./subPageA";
import "./subPageB";

import * as _ from "lodash";
console.log("At page 'A' :", _);

export default "pageA";
```

**`pageB.js`:**

```javascript
import "./subPageA";
import "./subPageB";

import * as _ from "lodash";
console.log("At page 'B' :", _);

export default "pageB";
```

好了，到此为止，需要编写的代码已经完成了。[>>> src 文件夹项目源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo03/src)

### 2. 编写`webpack`配置文件

首先我们应该安装先关的库，创建`package.json`，输入以下内容：

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

在命令行中运行`npm install`即可。

然后配置`webpack.config.js`文件。文件配置如下：

```javascript
const webpack = require("webpack");
const path = require("path");

module.exports = {
  // 多页面应用
  entry: {
    pageA: "./src/pageA.js",
    pageB: "./src/pageB.js"
  },
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "[name].bundle.js",
    chunkFilename: "[name].chunk.js"
  },
  optimization: {
    splitChunks: {
      cacheGroups: {
        // 注意: priority属性
        // 其次: 打包业务中公共代码
        common: {
          name: "common",
          chunks: "all",
          minSize: 1,
          priority: 0
        },
        // 首先: 打包node_modules中的文件
        vendor: {
          name: "vendor",
          test: /[\\/]node_modules[\\/]/,
          chunks: "all",
          priority: 10
        }
      }
    }
  }
};
```

着重来看`optimization.splitChunks`配置。我们将需要打包的代码放在`cacheGroups`属性中。

其中，每个键对值就是被打包的一部分。例如代码中的`common`和`vendor`。值得注意的是，针对第三方库（例如`lodash`）通过设置`priority`来让其先被打包提取，最后再提取剩余代码。

所以，上述配置中公共代码的提取顺序其实是：

```javascript
... ...
vendor: {
  name: "vendor",
  test: /[\\/]node_modules[\\/]/,
  chunks: "all",
  priority: 10
},
common: {
    name: "common",
    chunks: "all",
    minSize: 1,
    priority: 0
}
... ...
```

### 3. 打包和引用

命令行中运行`webpack`即可打包。可以看到，我们成功提取了公共代码，如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/1.png)

最后，打包的结果在`dist/`文件夹下面，我们要在`index.html`中引用打包好的`js`文件,`index.html`代码如下：

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
    <script src="./dist/common.chunk.js"></script>
    <script src="./dist/vendor.chunk.js"></script>
    <script src="./dist/pageA.bundle.js"></script>
    <script src="./dist/pageB.bundle.js"></script>
  </body>
</html>
```

使用 Chrome 或者 Firfox 打开`index.html`，并且打开控制台即可。
