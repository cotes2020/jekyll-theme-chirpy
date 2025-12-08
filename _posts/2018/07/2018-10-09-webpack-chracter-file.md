---
title: "十一：字体文件处理"
date: 2018-10-09
permalink: /2018-10-09-webpack-chracter-file/
categories: ["A开源技术课程", "webpack4系列教程"]
---

> 在自己的项目中引入中意的字体样式，是让自己舒坦的重要方式 :)

源码地址如下：

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo11)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

<!-- more -->

## 0. 课程介绍和资料

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo11)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课的代码目录如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/24.png)

本节课的`package.json`内容如下：

```json
{
  "devDependencies": {
    "css-loader": "^1.0.0",
    "extract-text-webpack-plugin": "^4.0.0-beta.0",
    "file-loader": "^1.1.11",
    "style-loader": "^0.21.0",
    "url-loader": "^1.0.1",
    "webpack": "^4.16.1"
  }
}
```

## 1. 准备字体文件和样式

如上面的代码目录所示，字体文件和样式都放在了`/src/assets/fonts/`目录下。[点我直接下载相关文件](https://github.com/dongyuanxin/webpack-demos/tree/master/demo11/src/assets/fonts)

## 2. 编写入口文件

为了提取 css 样式到单独文件，需要用到`ExtractTextPlugin`插件。在项目的入口文件需要引入`style-loader`和`css-loader`:

```javascript
// app.js
import "style-loader/lib/addStyles";
import "css-loader/lib/css-base";

import "./assets/fonts/iconfont.css";
```

## 3. 处理字体文件

借助`url-loader`，可以识别并且处理`eot`、`woff`等结尾的字体文件。同时，根据字体文件大小，可以灵活配置是否进行`base64`编码。下面的 demo 就是当文件大小小于`5000B`的时候，进行`base64`编码。

```javascript
// webpack.config.js

const path = require("path");
const ExtractTextPlugin = require("extract-text-webpack-plugin");

let extractTextPlugin = new ExtractTextPlugin({
  filename: "[name].min.css",
  allChunks: false
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
            loader: "style-loader"
          },
          use: [
            {
              loader: "css-loader"
            }
          ]
        })
      },
      {
        test: /\.(eot|woff2?|ttf|svg)$/,
        use: [
          {
            loader: "url-loader",
            options: {
              name: "[name]-[hash:5].min.[ext]",
              limit: 5000, // fonts file size <= 5KB, use 'base64'; else, output svg file
              publicPath: "fonts/",
              outputPath: "fonts/"
            }
          }
        ]
      }
    ]
  },
  plugins: [extractTextPlugin]
};
```

## 4. 编写`index.html`

按照上面的配置，打包好的`css`和`js`均位于`/src/dist/`文件夹下。因此，需要在`index.html`中引入这两个文件（假设已经打包完毕）：

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
    <div id="app">
      <div class="box">
        <i class="iconfont icon-xiazai"></i>
        <i class="iconfont icon-shoucang"></i>
        <i class="iconfont icon-erweima"></i>
        <i class="iconfont icon-xiangshang"></i>
        <i class="iconfont icon-qiehuanzuhu"></i>
        <i class="iconfont icon-sort"></i>
        <i class="iconfont icon-yonghu"></i>
      </div>
    </div>
    <script src="./dist/app.bundle.js"></script>
  </body>
</html>
```

## 5. 结果分析和验证

`CMD`中运行`webpack`进行打包，打包结果如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/25.png)

在 Chrome 中打开`index.html`，字体文件被正确引入：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/26.png)
