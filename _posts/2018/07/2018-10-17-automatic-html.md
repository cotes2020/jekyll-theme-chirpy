---
title: "十三：自动生成HTML文件"
date: 2018-10-17
permalink: /2018-10-17-automatic-html/
categories: ["A开源技术课程", "webpack4系列教程"]
---

> 在真实生产环境中，运行`webpack`进行打包后，完整的`index.html`应该是被自动生成的。例如静态资源、js 脚本都被自动插入了。而不是像之前的教程那样根据生成的文件手动插入。

为了实现这个功能，需要借助`HtmlWebpackPlugin`根据指定的`index.html`模板生成对应的 html 文件，还需要配合`html-loader`处理 html 文件中的`<img>`标签和属性。

<!-- more -->

## 0. 课程介绍和资料

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo13)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课的代码目录如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/30.png)

本节课用的 plugin 和 loader 的配置文件`package.json`如下：

```json
{
  "devDependencies": {
    "file-loader": "^1.1.11",
    "html-loader": "^0.5.5",
    "html-webpack-plugin": "^3.2.0",
    "url-loader": "^1.0.1",
    "webpack": "^4.16.1"
  }
}
```

## 1. 为什么要自动生成 HTML？

看过这个系列教程的朋友，都知道在之前的例子中，每次执行`webpack`打包生成`js`文件后，都必须在`index.html`中手动插入打包好的文件的路径。

但在真实生产环境中，一次运行`webpack`后，完整的`index.html`应该是被自动生成的。例如静态资源、js 脚本都被自动插入了。

为了实现这个功能，需要借助`HtmlWebpackPlugin`根据指定的`index.html`模板生成对应的 html 文件，还需要配合`html-loader`处理 html 文件中的`<img>`标签和属性。

## 2. 编写入口文件

编写`src/vendor/sum.js`文件，封装`sum()`函数作为示例，被其他文件引用（模块化编程）：

```javascript
export function sum(a, b) {
  return a + b;
}
```

编写入口文件`src/app.js`，引入上面编写的`sum()`函数，并且运行它，以方便我们在控制台检查打包结果：

```javascript
import { sum } from "./vendor/sum";

console.log("1 + 2 =", sum(1, 2));
```

## 3. 编写 HTML 文件

根目录下的`index.html`会被`html-webpack-plugin`作为最终生成的 html 文件的模板。打包后，相关引用关系和文件路径都会按照正确的配置被添加进去。

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
    <div></div>
    <img src="./src/assets/imgs/xunlei.png" />
  </body>
</html>
```

## 4. 编写`Webpack`配置文件

老规矩，`HtmlWebpackPlugin`是在`plugin`这个选项中配置的。常用参数含义如下：

- filename：打包后的 html 文件名称
- template：模板文件（例子源码中根目录下的 index.html）
- chunks：和`entry`配置中相匹配，支持多页面、多入口
- minify.collapseWhitespace：压缩选项

除此之外，因为我们在`index.html`中引用了`src/assets/imgs/`目录下的静态文件（图片类型）。需要用`url-loader`处理图片，然后再用`html-loader`声明。注意两者的处理顺序，`url-loader`先处理，然后才是`html-loader`处理。

```javascript
const path = require("path");
const webpack = require("webpack");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
  entry: {
    app: "./src/app.js"
  },
  output: {
    publicPath: __dirname + "/dist/",
    path: path.resolve(__dirname, "dist"),
    filename: "[name]-[hash:5].bundle.js",
    chunkFilename: "[name]-[hash:5].chunk.js"
  },
  module: {
    rules: [
      {
        test: /\.html$/,
        use: [
          {
            loader: "html-loader",
            options: {
              attrs: ["img:src"]
            }
          }
        ]
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
          }
        ]
      }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      filename: "index.html",
      template: "./index.html",
      chunks: ["app"], // entry中的app入口才会被打包
      minify: {
        // 压缩选项
        collapseWhitespace: true
      }
    })
  ]
};
```

## 5. 结果和测试

运行`webpack`进行打包，下面是打包结果：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/33.png)

可以在`/dist/`中查看自动生成的`index.html`文件，如下图所示，脚本和静态资源路径都被正确处理了：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/31.png)

直接在 Chrome 打开`index.html`，并且打开控制台：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/32.png)

图片成功被插入到页面，并且 js 的运行也没有错误，成功。

## 6. 更多资料

- `html-loader`文档: [https://www.webpackjs.com/loaders/html-loader/](https://www.webpackjs.com/loaders/html-loader/)
- `html-webpack-plugin`文档: [https://www.webpackjs.com/plugins/html-webpack-plugin/](https://www.webpackjs.com/plugins/html-webpack-plugin/)
