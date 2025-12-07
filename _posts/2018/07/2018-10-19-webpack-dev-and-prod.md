---
title: "十六：开发模式和生产模式·实战"
date: 2018-10-19
permalink: /2018-10-19-webpack-dev-and-prod/
categories: ["开源技术课程", "webpack4系列教程"]
---

> 这是`webpack4`系列最后一篇教程了。这篇文章在之前所有教程的基础上，做了一个真正意义上的 webpack 项目！我花费了三个月整理了这份教程，并且完善了相关示例代码，也更熟悉 webpack 的理论和应用，当然，也感谢大家的支持。

好了，感慨完毕，开始正题 ?

<!-- more -->

> [作者按](https://godbmw.com):这是`webpack4`系列最后一篇教程了。这篇文章在之前所有教程的基础上，做了一个真正意义上的 webpack 项目！我花费了三个月整理了这份教程，并且完善了相关示例代码，也更熟悉 webpack 的理论和应用，当然，也感谢大家的支持。好了，感慨完毕，开始正题 👇

## 0. 课程介绍和资料

- [>>>本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo16)
- [>>>所有课程源码](https://github.com/dongyuanxin/webpack-demos)

本节课的代码目录如下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/42.png)

## 1. 如何分离开发环境和生产环境？

熟悉 Vuejs 或者 ReactJs 的脚手架的朋友应该都知道：在根目录下有一个`/build/`文件夹，专门放置`webpack`配置文件的相关代码。

不像我们前 15 节课的 demo (只有一个配置文件`webpack.config.js`)，**为了分离开发环境和生产环境，我们需要分别编写对应的`webpack`配置代码。**

毫无疑问，有一些插件和配置是两种环境共用的，所以应该提炼出来，避免重复劳动。如前文目录截图，`build/webpack.common.conf.js`就保存了两种环境都通用的配置文件。而`build/webpack.dev.conf.js`和`build/webpack.prod.conf.js`分别是开发和生产环境需要的特殊配置。

## 2. 编写`package.json`

类似上一节讲的，为了让命令更好调用，需要配置`scripts`选项。模仿`vue-cli`的命令格式，编写如下`package.json`:

```json
{
  "scripts": {
    "dev": "webpack-dev-server --env development --open --config build/webpack.common.conf.js",
    "build": "webpack --env production --config build/webpack.common.conf.js"
  },
  "devDependencies": {
    "babel-core": "^6.26.3",
    "babel-loader": "^7.1.5",
    "babel-plugin-transform-runtime": "^6.23.0",
    "babel-preset-env": "^1.7.0",
    "clean-webpack-plugin": "^0.1.19",
    "css-loader": "^1.0.0",
    "extract-text-webpack-plugin": "^4.0.0-beta.0",
    "html-webpack-plugin": "^3.2.0",
    "jquery": "^3.3.1",
    "style-loader": "^0.21.0",
    "webpack": "^4.16.1",
    "webpack-cli": "^3.1.0",
    "webpack-dev-server": "^3.1.4",
    "webpack-merge": "^4.1.3"
  },
  "dependencies": {
    "babel-polyfill": "^6.26.0",
    "babel-runtime": "^6.26.0"
  }
}
```

按照配置，运行：

- `npm run dev`: 进入开发调试模式
- `npm run build`: 生成打包文件

还可以看出来，`build/webpack.common.conf.js`不仅仅是存放着两种环境的公共代码，还是`webpack`命令的入口文件。

## 3. 如何合并 webpack 的不同配置？

根据前面所讲，我们有 3 个配置文件。那么如何在`build/webpack.common.conf.js`中引入开发或者生产环境的配置，并且正确合并呢？

此时需要借助`webpack-merge`这个第三方库。下面是个示例代码：

```javascript
const merge = require("webpack-merge");

const productionConfig = require("./webpack.prod.conf");
const developmentConfig = require("./webpack.dev.conf");

const commonConfig = {}; // ... 省略

module.exports = env => {
  let config = env === "production" ? productionConfig : developmentConfig;
  return merge(commonConfig, config); // 合并 公共配置 和 环境配置
};
```

## 4. 如何在代码中区分不同环境？

### 4.1 配置文件

如果这个 js 文件是 webpack 命令的入口文件，例如`build/webpack.common.conf.js`，那么`mode`的值（production 或者 development）会被自动传入`module.exports`的第一个参数，开发者可以直接使用。

如下面的代码，先判断是什么环境，然后再决定使用什么配置，最后 return 给 webpack：

```javascript
module.exports = env => {
  let config = env === "production" ? productionConfig : developmentConfig;
  return merge(commonConfig, config); // 合并 公共配置 和 环境配置
};
```

### 4.2 项目文件

如果这个 js 文件是项目中的脚本文件，那么可以访问`process.env.NODE_ENV`这个变量来判断环境：

```javascript
if (process.env.NODE_ENV === "development") {
  console.log("开发环境");
} else {
  console.log("生产环境");
}
```

## 5. 编写配置文件

### 5.1 编写公共配置文件

```javascript
// /build/webpack.common.conf.js

const webpack = require("webpack");
const merge = require("webpack-merge");
const ExtractTextPlugin = require("extract-text-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");

const path = require("path");

const productionConfig = require("./webpack.prod.conf.js"); // 引入生产环境配置文件
const developmentConfig = require("./webpack.dev.conf.js"); // 引入开发环境配置文件

/**
 * 根据不同的环境，生成不同的配置
 * @param {String} env "development" or "production"
 */
const generateConfig = env => {
  // 将需要的Loader和Plugin单独声明

  let scriptLoader = [
    {
      loader: "babel-loader"
    }
  ];

  let cssLoader = [
    {
      loader: "css-loader",
      options: {
        minimize: true,
        sourceMap: env === "development" ? true : false // 开发环境：开启source-map
      }
    }
  ];

  let styleLoader =
    env === "production"
      ? ExtractTextPlugin.extract({
          // 生产环境：分离、提炼样式文件
          fallback: {
            loader: "style-loader"
          },
          use: cssLoader
        })
      : // 开发环境：页内样式嵌入
        cssLoader;

  return {
    entry: { app: "./src/app.js" },
    output: {
      publicPath: env === "development" ? "/" : __dirname + "/../dist/",
      path: path.resolve(__dirname, "..", "dist"),
      filename: "[name]-[hash:5].bundle.js",
      chunkFilename: "[name]-[hash:5].chunk.js"
    },
    module: {
      rules: [
        { test: /\.js$/, exclude: /(node_modules)/, use: scriptLoader },
        { test: /\.css$/, use: styleLoader }
      ]
    },
    plugins: [
      // 开发环境和生产环境二者均需要的插件
      new HtmlWebpackPlugin({
        filename: "index.html",
        template: path.resolve(__dirname, "..", "index.html"),
        chunks: ["app"],
        minify: {
          collapseWhitespace: true
        }
      }),
      new webpack.ProvidePlugin({ $: "jquery" })
    ]
  };
};

module.exports = env => {
  let config = env === "production" ? productionConfig : developmentConfig;
  return merge(generateConfig(env), config);
};
```

### 5.2 编写开发环境配置文件

```javascript
// /build/webpack.dev.conf.js

const webpack = require("webpack");

const path = require("path");

module.exports = {
  mode: "development",
  devtool: "source-map",
  devServer: {
    contentBase: path.join(__dirname, "../dist/"),
    port: 8000,
    hot: true,
    overlay: true,
    proxy: {
      "/comments": {
        target: "https://m.weibo.cn",
        changeOrigin: true,
        logLevel: "debug",
        headers: {
          Cookie: ""
        }
      }
    },
    historyApiFallback: true
  },
  plugins: [
    new webpack.HotModuleReplacementPlugin(),
    new webpack.NamedModulesPlugin()
  ]
};
```

### 5.3 编写生产环境配置文件

```javascript
// /build/webpack.comm.conf.js

const ExtractTextPlugin = require("extract-text-webpack-plugin");
const CleanWebpackPlugin = require("clean-webpack-plugin");

const path = require("path");

module.exports = {
  mode: "production",
  plugins: [
    new ExtractTextPlugin({
      filename: "[name].min.css",
      allChunks: false // 只包括初始化css, 不包括异步加载的CSS
    }),
    new CleanWebpackPlugin(["dist"], {
      root: path.resolve(__dirname, "../"),
      verbose: true
    })
  ]
};
```

## 6. 其他文件

在项目目录截图中展示的样式文件，vendor 下的文件还有 app.js，代码就不一一列出了。完全可以根据自己的需要，写一些简单的代码，然后运行一下。毕竟前面的配置文件的架构和讲解才是最重要的。

这里仅仅给出源码地址（欢迎 Star 哦）：

- 入口文件`/src/app.js`：[https://github.com/dongyuanxin/webpack-demos/blob/master/demo16/src/app.js](https://github.com/dongyuanxin/webpack-demos/blob/master/demo16/src/app.js)
- `/src/style/`下的所有样式文件：[https://github.com/dongyuanxin/webpack-demos/tree/master/demo16/src/style](https://github.com/dongyuanxin/webpack-demos/tree/master/demo16/src/style)
- `/src/vendor/`下的所有脚本文件：[https://github.com/dongyuanxin/webpack-demos/tree/master/demo16/src/vendor](https://github.com/dongyuanxin/webpack-demos/tree/master/demo16/src/vendor)

## 7. 运行效果和测试

鼓捣这么半天，肯定要测试下，要不怎么才能知道正确性（_这才是另人激动的一步啦啦啦_）。

### 7.1 跑起来：开发模式

进入项目目录，运行`npm run dev`:

![](https://static.godbmw.com/images/webpack/webpack4系列教程/43.png)

成功跑起来，没出错（废话，都是被调试了好多次了哈哈哈）。

打开浏览器的控制台看一下：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/44.png)

很好，都是按照编写的`app.js`的逻辑输出的。

### 7.2 跑起来：生产模式

按`Ctrl+C`退出开发模式后，运行`npm run build`，如下图打包成功：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/45.png)

打包后的文件也放在了指定的位置：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/46.png)

直接点击`index.html`，并且打开浏览器控制台：

![](https://static.godbmw.com/images/webpack/webpack4系列教程/47.png)

ok, 符合`app.js`的输出：成功辨识了是否是开发环境！！！

## 8. 最终

**完结撒花 ✿✿ ヽ(°▽°)ノ ✿**
