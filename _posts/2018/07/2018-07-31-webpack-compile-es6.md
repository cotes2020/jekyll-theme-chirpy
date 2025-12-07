---
title: "二：编译ES6"
date: 2018-07-31
permalink: /2018-07-31-webpack-compile-es6/
categories: ["开源技术课程", "webpack4系列教程"]
---

> 今天介绍`webpack`怎么编译`ES6`的各种函数和语法。敲黑板：**这是`webpack4`版本哦, 有一些不同于`webpack3`的地方。**

[>>> 本节课源码](https://github.com/dongyuanxin/webpack-demos/tree/master/demo02)

[>>> 所有课程源码](https://github.com/dongyuanxin/webpack-demos)

### 1. 了解`babel`

说起编译`es6`，就必须提一下`babel`和相关的技术生态：

1. `babel-loader`: 负责 es6 语法转化
2. `babel-preset-env`: 包含 es6、7 等版本的语法转化规则
3. `babel-polyfill`: es6 内置方法和函数转化垫片
4. `babel-plugin-transform-runtime`: 避免 polyfill 污染全局变量

需要注意的是, `babel-loader`和`babel-polyfill`。前者负责语法转化，比如：箭头函数；后者负责内置方法和函数，比如：`new Set()`。

### 2. 安装相关库

这次，我们的`package.json`文件配置如下：

```javascript
{
  "devDependencies": {
    "babel-core": "^6.26.3",
    "babel-loader": "^7.1.5",
    "babel-plugin-transform-runtime": "^6.23.0",
    "babel-preset-env": "^1.7.0",
    "webpack": "^4.15.1"
  },
  "dependencies": {
    "babel-polyfill": "^6.26.0",
    "babel-runtime": "^6.26.0"
  }
}
```

[>>> package.json 配置地址](https://github.com/dongyuanxin/webpack-demos/blob/master/demo02/package.json)

### 3. `webpack`中使用`babel`

> `babel`的相关配置，推荐单独写在`.babelrc`文件中。下面，我给出这次的相关配置：

```javascript
{
  "presets": [
    [
      "env",
      {
        "targets": {
          "browsers": ["last 2 versions"]
        }
      }
    ]
  ],
  "plugins": ["transform-runtime"]
}
```

在`webpack`配置文件中，关于`babel`的调用需要写在`module`模块中。对于相关的匹配规则，除了匹配`js`结尾的文件，还应该去除`node_module/`文件夹下的第三库的文件（发布前已经被处理好了）。

```javascript
module.exports = {
  entry: {
    app: "./app.js"
  },
  output: {
    filename: "bundle.js"
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules)/,
        use: {
          loader: "babel-loader"
        }
      }
    ]
  }
};
```

[>>> .babelrc 地址](https://github.com/dongyuanxin/webpack-demos/blob/master/demo02/.babelrc)

[>>> 配置文件地址](https://github.com/dongyuanxin/webpack-demos/blob/master/demo02/webpack.config.js)

### 4. 最后：`babel-polyfill`

我们发现整个过程中并没有使用`babel-polyfill`。**它需要在我们项目的入口文件中被引入**，或者在`webpack.config.js`中配置。这里我们采用第一种方法编写`app.js`:

```javascript
import "babel-polyfill";
let func = () => {};
const NUM = 45;
let arr = [1, 2, 4];
let arrB = arr.map(item => item * 2);

console.log(arrB.includes(8));
console.log("new Set(arrB) is ", new Set(arrB));
```

命令行中进行打包，然后编写`html`文件引用打包后的文件即可在不支持`es6`规范的老浏览器中看到效果了。

### 5. 相关资料

- [polyfill 引入](https://www.babeljs.cn/docs/usage/polyfill/)：https://www.babeljs.cn/docs/usage/polyfill/
- [babel-preset-env 配置](https://www.babeljs.cn/docs/plugins/preset-env/)：https://www.babeljs.cn/docs/plugins/preset-env/
