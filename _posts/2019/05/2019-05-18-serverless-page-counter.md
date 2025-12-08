---
title: "Serverless开发一款极简网页计数器"
date: 2019-05-18
permalink: /2019-05-18-serverless-page-counter/
categories: ["C工作实践分享"]
---
![007S8ZIlgy1giwwzo7rfaj310f0oy753.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-05-18-serverless-page-counter/007S8ZIlgy1giwwzo7rfaj310f0oy753.jpg)


这几天基于支持 HTML5 无感认证的 ServerLess 平台开发了一款博客、门户网站等 web 平台常用的 PV 统计工具：[page-counter](https://github.com/dongyuanxin/page-counter) 。主要用到的技术是 js+webpack。


回首看来，解决了以下几个比较有意思的问题：

- 如何设计代码，用统一的方式支持多个 ServerLess 平台？
- 如何架构项目，使得其支持 CDN 和 npm 两种方式引入？
- 如何精简源码，源码大小控制在 4kb？
- 如何借助 webpack 分离生产和测试环境？

源码地址：[https://github.com/dongyuanxin/page-counter](https://github.com/dongyuanxin/page-counter)<br />npm 地址：[https://www.npmjs.com/package/page-counter](https://www.npmjs.com/package/page-counter)


如果有兴趣的同学，欢迎在阅读完本文后一起接入其他平台的开发； **觉得不错的同学，欢迎给个 Star 哦** 。


## 项目目录


![007S8ZIlgy1giwx0bbk2uj309m0jywfd.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2019-05-18-serverless-page-counter/007S8ZIlgy1giwx0bbk2uj309m0jywfd.jpg)


如上图所示，bin/backend 目录是暂时没用的。几个比较重要的目录功能说明：

- `build/` : webpack 的配置文件，分别是公共配置、开发模式配置、生产模式配置
- `dist/`
	- index.template.html: 开发模式下配合 webpack 的 html 模板文件
	- page-counter.min.js: 打包后的 page-counter 内容，供 CDN 引入
	- page-counter.bomb-1.6.7.min.js：我手动修改并且打包的 Bomb 平台源码
- `examples/` : gh-pages 页面，请看[此页面](https://godbmw.com/page-counter/)
- `deploy.sh` : gh-pages 部署脚本，支持 ssh 和 https 协议
- `index.js` : npm 的入口文件
- `index.build.js` : CDN 打包入口文件
- `src/` :
	- `serverless/` : 暴露不同平台的统一接口
	- `config.js` : 自动读取全局配置
	- `utils.js` : 常用函数方法

## 抽象接口：支持多 Serverless 平台


`src/serverless/interface.js`  中定义了不同平台的类的公共父类。虽然 js 不支持抽象接口，但是也可以通过抛出错误来实现：


```typescript
export default class ServerLessInterface {
    constructor() {}
    ACL() {
        throw new Error('Interface method "ACL" must be rewritten');
    }
    setData() {
        throw new Error('Interface method "setData" must be rewritten');
    }
    count() {
        throw new Error('Interface method "count" must be rewritten');
    }
}

```


而 leancloud.js 、bomb.js 等不同平台的类都要实现这个接口中的这 3 个方法。然后通过 `src/serverless/index.js`  统一暴露出去：


```typescript
import LeanCloud from "./leancloud";
import Bomb from "./bomb";
class ServerLessFactory {
    constructor(name) {
        name = name.toLocaleLowerCase();
        switch (name) {
            case "leancloud":
                return new LeanCloud();
            case "bomb":
                return new Bomb();
            default:
                throw new Error(
                    "Serverless must be one of [ leancloud, bomb ]"
                );
        }
    }
}
export default ServerLessFactory;
```


这两种设计，既解耦了不同平台的代码，而且还约束了实现规则。如果想接入更多平台，只需要创建新文件，并且暴露一个继承 `ServerLessInterface`  接口的指定方法的子类即可。


## 快速方便：支持 CDN 和 npm 的使用


一个成熟的前端小工具需要考虑到多种引入方式，目前主流的就是 cdn 和 npm。例如 jquery，cdn 引入，jq 会被自动挂载在 window 对象上；npm 引入，则作用域只在当前文件模块有用。


在考察了多种同类工具后，针对 cdn 和 npm 做了不同的处理。


**npm** <br />对外暴露 `PageCounter`  对象，其上有 3 个方法：

- `setData()` ：将当前页面信息发送到云数据库
- `countTotal()` : 统计数据库总记录数（网站总 PV），并且将返回结果**自动放入**id 为 page-counter-total-times 的标签里
- `countSingle()` : 统计数据库符合要求的记录数（当前页面 PV），并且将返回结果**自动放入**id 为 page-counter-single-times 的标签里

```typescript
import PageCounter from "./src";
export default PageCounter;
```


**CDN**不会在全局挂载上述对象方法，会自动执行上面的 3 种方法。考虑到并发以及 pv 数允许 1 以内的误差，没有保证串行。


```typescript
import PageCounter from "./src";
PageCounter.setData();
PageCounter.countTotal();
PageCounter.countSingle();
```


## 精简源码：巧用 package.json 和第三方 SDK


经过精简，打包后 cdn 引入的源码只有 4kb。npm 引入的话，webpack 会自动进行 tree shaking。因为要对接不同的 serverless 平台，因此需要使用他们的 sdk。


而这些 sdk 分成 2 种：

- 类似 leancloud：既可以 npm 引入，也可以 cdn 引入后自动挂载到 window 对象
- 类似 bomb：无 cdn 引入，只要 npm 引入

针对第二种情况，我采取的方案是手动打包编译。比如对于 bomb 的 sdk，专门创建新的工程，然后配合 webpack 和以下代码，进行打包。


```text
import Bomb from "hydrogen-js-sdk";
window.Bomb = Bomb;

```


打包后的源码放入版本库，这样借助  [https://unpkg.com](https://unpkg.com/)  等常见的 CDN 平台就可以引入了。这么做的很重要的一点是： **代码中都是通过 window 上的对象读取对应 serverless 平台的 api，这样就不会被 webpack 识别，进而发生重复无用打包** 。


关于读取配置的文件，都放在了 `src/config.js`  下。考虑到 script 标签引入造成的变量挂载时间点不确定，读取采用了动态读取。**为了操作起来更方便，而不是像调用函数那样，借助了 es6 类语法中的 setter 和 getter。**


```typescript
// 举个例子：
class Config {
    constructor() {}
    get serverless() {
        if (!window.PAGE_COUNTER_CONFIG) {
            throw new Error("Please init variable window.PAGE_COUNTER_CONFIG");
        }
        return window.PAGE_COUNTER_CONFIG.serverless || "leancloud";
    }
}
const config = new Config();
console.log(config.serverless); // 返回当前最新的window.PAGE_COUNTER_CONFIG.serverless

```


最后讲讲 package.json 的小技巧。虽然代码中没有使用 import 语法读取 sdk 的对象，但是我还是把 leancloud、bomb 平台的 sdk 放入了 `dependencies` 。这样做有什么好处呢？


用户只需要安装 page-counter 即可，其他 sdk 自动安装（不需要手动再敲命令）。然后用户就可以使用下面语法美滋滋引入：


```typescript
import("hydrogen-js-sdk")
    .then(res => {
        // 将 Bomb 对象挂载在 window 上
        window.Bomb = res.default;
        // 设置应用信息
        window.PAGE_COUNTER_CONFIG = {
            // ...
        };
        return import("page-counter");
    })
    .then(res => {
        const PageCounter = res.default;
        PageCounter.setData(); // 发送当前页面数据
        PageCounter.countTotal(); // 将总浏览量放入 ID 为 page-counter-total-times 的DOM元素中
        PageCounter.countSingle(); // 将当前页面浏览量放入 ID 为 page-counter-single-times 的DOM元素中
    });
```


## Webpack：分离生产和开发环境


不得不说，webpack 真的好用呀。脏活累活以及常见工具，它都给你承包了。


`webpack.base.conf.js`  是两种模式的公共配置，指明入口文件以及代码环境（web）。并且能够识别模式，然后自行拼接配置。


`webpack.prod.conf.js` ：生产模式，主要为了打包源码，方便 CDN 引入。


`webpack.dev.conf.js` : 开启热更新以及本地服务器方便调试，渲染的前端调试页面的模板文件就是 `dist/index.template.html` 。


