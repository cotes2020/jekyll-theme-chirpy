---
title: "十：图片处理汇总"
date: 2018-09-11
permalink: /2018-09-11-webpack-image/
categories: ["开源技术课程", "webpack4系列教程"]
---

本节课会讲述`webpack4`中的图片常用的基础操作：

- 图片处理 和 `Base64`编码
- 图片压缩
- 合成雪碧图

<!-- more -->

### 0. 课程源码和资料

本次课程的代码目录(如下图所示)：
![](https://static.godbmw.com/images/webpack/webpack4系列教程/19.png)

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo10)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课会讲述`webpack4`中的图片常用的基础操作：

- 图片处理 和 `Base64`编码
- 图片压缩
- 合成雪碧图

### 1. 准备工作

如项目代码目录展示的那样，除了常见的`app.js`作为入口文件，我们将用到的 3 张图片放在`/src/assets/img/`目录下，并在样式文件`base.css`中引用这些图片。

剩下的内容交给`webpack`打包处理即可。样式文件和入口 js 文件的代码分别如下所示：

```css
/* base.css */
*,
body {
  margin: 0;
  padding: 0;
}
.box {
  height: 400px;
  width: 400px;
  border: 5px solid #000;
  color: #000;
}
.box div {
  width: 100px;
  height: 100px;
  float: left;
}
.box .ani1 {
  background: url("./../assets/imgs/1.jpg") no-repeat;
}
.box .ani2 {
  background: url("./../assets/imgs/2.jpg") no-repeat;
}
.box .ani3 {
  background: url("./../assets/imgs/3.png") no-repeat;
}
```

```javascript
// app.js
import "style-loader/lib/addStyles";
import "css-loader/lib/css-base";

import "./css/base.css";
```

在处理图片和进行`base64`编码的时候，需要使用`url-loader`。

在压缩图片的时候，要使用`img-loader`插件，并且针对不同的图片类型启用不同的子插件。

而`postcss-loader`和`postcss-sprites`则用来合成雪碧图，减少网络请求。

因此，在 npm 安装完相关插件后，`package.json`的内容如下所示：

```json
{
  "devDependencies": {
    "css-loader": "^1.0.0",
    "extract-text-webpack-plugin": "^4.0.0-beta.0",
    "file-loader": "^1.1.11",
    "imagemin": "^5.3.1",
    "imagemin-pngquant": "^5.1.0",
    "img-loader": "^3.0.0",
    "postcss-loader": "^2.1.6",
    "postcss-sprites": "^4.2.1",
    "style-loader": "^0.21.0",
    "url-loader": "^1.0.1",
    "webpack": "^4.16.1"
  }
}
```

同时，我们编写如下`index.html`(假设已经打包好了项目文件，现在直接引入)：

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
        <div class="ani1"></div>
        <div class="ani2"></div>
        <div class="ani3"></div>
      </div>
    </div>
    <script src="./dist/app.bundle.js"></script>
  </body>
</html>
```

### 2. 图片处理 和 Base64 编码

#### 2.1 webpack 配置

为了方便样式提取，还是利用`extract-text-webpack-plugin`来提取样式文件。

同时，在`module.rules`选项中进行配置，以实现让 loader 识别图片后缀名，并且进行指定的处理操作。

代码如下：

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
        test: /\.(png|jpg|jpeg|gif)$/,
        use: [
          {
            loader: "url-loader",
            options: {
              name: "[name]-[hash:5].min.[ext]",
              limit: 20000, // size <= 20KB
              publicPath: "static/",
              outputPath: "static/"
            }
          }
        ]
      }
    ]
  },
  plugins: [extractTextPlugin]
};
```

通过配置`url-loader`的 limit 选项，可以根据图片大小来决定是否进行`base64`编码。这次配置的是：小于 20kb 的图片进行`base64`编码。

#### 2.2 打包结果

之前提到过，在项目中引入了 3 张图片，其中`3.png`是小于 20kb 的图片。在命令行中运行`webpack`进行打包，size 小于 20kb 的图片被编码，只打包了 2 个 size 大于 20kb 的图片文件：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/20.png)

打开浏览器的控制台，我们的图片已经被成功编码：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/21.png)

### 3. 图片压缩

#### 3.1 压缩配置

图片压缩需要使用`img-loader`，除此之外，针对不同的图片类型，还要引用不同的插件。比如，我们项目中使用的是 png 图片，因此，需要引入`imagemin-pngquant`，并且指定压缩率。

我们只需要在上面的配置文件中将下方代码：

```javascript
// ...
{
  test: /\.(png|jpg|jpeg|gif)$/,
  use: [
    {
      loader: "url-loader",
      options: {
        name: "[name]-[hash:5].min.[ext]",
        limit: 20000, // size <= 20KB
        publicPath: "static/",
        outputPath: "static/"
      }
    }
  ]
}
// ...
```

替换为下方代码即可，因为执行顺序问题，我们将 url-loader 的 limit 设置成 1kb，来防止压缩后的 png 图片被 base64 编码：

```javascript
// ...
{
  test: /\.(png|jpg|jpeg|gif)$/,
  use: [
    {
      loader: "url-loader",
      options: {
        name: "[name]-[hash:5].min.[ext]",
        limit: 1000, // size <= 1KB
        publicPath: "static/",
        outputPath: "static/"
      }
    },
    // img-loader for zip img
    {
      loader: "img-loader",
      options: {
        plugins: [
          require("imagemin-pngquant")({
            quality: "80" // the quality of zip
          })
        ]
      }
    }
  ]
}
// ...
```

#### 3.2 打包结果

运行 webpack 打包，查看打包结果：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/22.png)

是的，如你所见，10.5kb 大小的迅雷图标，被压缩到了 1.8kb。图片信息可以去 github 上查看，在文章开头有提及 github 地址。

#### 3.3 遗留问题

并没有解决`jpg`格式图片压缩。根据[`img-loader`的官方文档](https://www.npmjs.com/package/img-loader)，安装了`imagemin-mozjpeg`插件。

但是这个插件的最新版本是`7.0.0`，然而配置后，webpack 启动会用报错。

查看了 github 上的 issue，我将版本回退到`6.0.0`。可以安装，也可以配置运行，正常打包。但是打包后的 jpg 图片大小并没有变化，也就是说，并没有被压缩！！！

**希望有大佬可以指点一下小生，万分感谢**

### 4. 合成雪碧图

#### 4.1 webpack 配置

在之前的基础上，配置还是很简单的，loader 的引入和环境变量都在注释里面了：

```javascript
const path = require("path");
const ExtractTextPlugin = require("extract-text-webpack-plugin");

let extractTextPlugin = new ExtractTextPlugin({
  filename: "[name].min.css",
  allChunks: false
});

/*********** sprites config ***************/
let spritesConfig = {
  spritePath: "./dist/static"
};
/******************************************/

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
            },
            /*********** loader for sprites ***************/
            {
              loader: "postcss-loader",
              options: {
                ident: "postcss",
                plugins: [require("postcss-sprites")(spritesConfig)]
              }
            }
            /*********************************************/
          ]
        })
      },
      {
        test: /\.(png|jpg|jpeg|gif)$/,
        use: [
          {
            loader: "url-loader",
            options: {
              name: "[name]-[hash:5].min.[ext]",
              limit: 10000, // size <= 20KB
              publicPath: "static/",
              outputPath: "static/"
            }
          },
          {
            loader: "img-loader",
            options: {
              plugins: [
                require("imagemin-pngquant")({
                  quality: "80"
                })
              ]
            }
          }
        ]
      }
    ]
  },
  plugins: [extractTextPlugin]
};
```

#### 4.2 效果展示

按照我们的配置，打包好的雪碧图被放入了`/dist/static/`目录下，如下图所示：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/23.png)

#### 4.3 雪碧图的实际应用

雪碧图是为了减少网络请求，所以被处理雪碧图的图片多为各式各样的 logo 或者大小相等的小图片。而对于大图片，还是不推荐使用雪碧图。

除此之外，雪碧图要配合 css 代码进行定制化使用。要通过 css 代码在雪碧图上精准定位需要的图片（可以理解成从雪碧图上裁取需要的图片），更多可以百度或者 google。
