---
title: "五：处理CSS"
date: 2018-08-17
permalink: /2018-08-17-webpack-css/
categories: ["A开源技术课程", "webpack4系列教程"]
---

本节课结合`webpack`和相关 LOADER 的特点，可以非常方便地处理 CSS。主要包括以下 4 个部分：

1. 将 css 通过 link 标签引入
2. 将 css 放在 style 标签里
3. 动态卸载和加载 css
4. 页面加载 css 前的`transform`

将配合源码逐一演示讲解。

<!-- more -->

> 这节课讲解`webpack4`中打包`css`的应用。v4 版本和 v3 版本并没有特别的出入。

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo05)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

### 1. 准备工作

众所周知，CSS 在 HTML 中的常用引入方法有`<link>`标签和`<style>`标签两种，所以这次就是结合`webpack`特点实现以下功能：

1. 将 css 通过 link 标签引入
2. 将 css 放在 style 标签里
3. 动态卸载和加载 css
4. 页面加载 css 前的`transform`

下图展示了这次的目录代码结构：
![](https://static.godbmw.com/images/webpack/webpack4系列教程/7.png)

这次我们需要用到`css-loader`，`file-loader`等 LOADER，`package.json`如下：

```json
{
  "devDependencies": {
    "css-loader": "^1.0.0",
    "file-loader": "^1.1.11",
    "style-loader": "^0.21.0"
  }
}
```

其中，`base.css`代码如下：

```css
*,
body {
  margin: 0;
  padding: 0;
}
html {
  background: red;
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

### 2. `CSS`通过`<link>`标签引入

> link 标签通过引用 css 文件，所以需要借助`file-loader`来将 css 处理为文件。

`webpack.config.js`:

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
        test: /\.css$/, // 针对CSS结尾的文件设置LOADER
        use: [
          {
            loader: "style-loader/url"
          },
          {
            loader: "file-loader"
          }
        ]
      }
    ]
  }
};
```

为了让效果更显著，编写如下`app.js`:

```javascript
let clicked = false;
window.addEventListener("click", function() {
  // 需要手动点击页面才会引入样式！！！
  if (!clicked) {
    import("./css/base.css");
  }
});
```

### 3. `CSS`放在`<style>`标签里

> 通常来说，`css`放在`style`标签里可以减少网络请求次数，提高响应时间。需要注意的是，_在老式 IE 浏览器中，对`style`标签的数量是有要求的_。

`app.js`和第二部分一样，`webpack.config.js`配置修改如下：

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
        test: /\.css$/, // 针对CSS结尾的文件设置LOADER
        use: [
          {
            loader: "style-loader",
            options: {
              singleton: true // 处理为单个style标签
            }
          },
          {
            loader: "css-loader",
            options: {
              minimize: true // css代码压缩
            }
          }
        ]
      }
    ]
  }
};
```

### 4. 动态卸载和加载`CSS`

> `style-loader`为 css 对象提供了`use()`和`unuse()`两种方法，借助这两种方法，可以方便快捷地加载和卸载 css 样式。

首先，需要配置`webpack.config.js`:

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
        test: /\.css$/,
        use: [
          {
            loader: "style-loader/useable" // 注意此处的style-loader后面的 useable
          },
          {
            loader: "css-loader"
          }
        ]
      }
    ]
  }
};
```

然后，我们修改我们的`app.js`，来实现每 0.5s 换一次背景颜色：

```javascript
import base from "./css/base.css"; // import cssObj from '...'
var flag = false;
setInterval(function() {
  // unuse和use 是 cssObj上的方法
  if (flag) {
    base.unuse();
  } else {
    base.use();
  }
  flag = !flag;
}, 500);
```

打包后打开`index.html`即可看到页面背景颜色闪动的效果。

### 5. 页面加载`css`前的`transform`

> 对于`css`的`transform`，简单来说：**在加载 css 样式前，可以更改 css**。这样，方便开发者根据业务需要，对 css 进行相关处理。

需要对`style-loader`增加`options.transform`属性，值为指定的 js 文件，所以, `webpack.config.js`配置如下：

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
        test: /\.css$/,
        use: [
          {
            loader: "style-loader",
            options: {
              transform: "./css.transform.js" // transform 文件
            }
          },
          {
            loader: "css-loader"
          }
        ]
      }
    ]
  }
};
```

下面，我们编写`css.transform.js`，这个文件导出一个函数，传入的参数就是 css 字符串本身。

```javascript
module.exports = function(css) {
  console.log(css); // 查看css
  return window.innerWidth < 1000 ? css.replace("red", "green") : css; // 如果屏幕宽度 < 1000, 替换背景颜色
};
```

在`app.js`中引入 css 文件即可：

```javascript
import base from "./css/base.css";
```

我们打开控制台，如下图所示，当屏幕宽度小于 1000 时候，css 中的`red`已经被替换为了`green`。

![](https://static.godbmw.com/images/webpack/webpack4系列教程/8.png)

需要注意的是：`transform`是在 css 引入前根据需要修改，所以之后是不会改变的。所以上方代码不是响应式，当把屏幕宽度拉长到大于 1000 时候，依旧是绿色。重新刷新页面，才会是红色。
