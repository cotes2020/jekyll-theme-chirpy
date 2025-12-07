---
title: "实战篇：当Koa遇上Typescript的时候"
date: "2019-08-27"
permalink: /2019-08-27-koa-meet-typescript/
---

最近在做运营侧中台项目的重构，目前的选型是 koa2+typescript。在实际生产中，切实体会到了 typescript 类型带来的好处。

为了更形象说明 typescript 的优势，还是先来看一个场景吧：

## BUG 现场

作为一门灵活度特别大的语言，坏处就是：**复杂逻辑编写过程中，数据结构信息可能由于逻辑复杂、人员变更等情况而丢失，从而写出来的代码含有隐含错误**。

比如这次我在给[自己的博客](https://github.com/dongyuanxin/blog/)编写[node 脚本](https://github.com/dongyuanxin/blog/blob/master/bin/prettier.js)的时候就遇到了这种情况：

```javascript
const result = [];

function findAllFiles(root) {
  const files = fs.readdirSync(root);
  files.forEach(name => {
    const file = path.resolve(root, name);
    if (isFolder(file)) {
      findAllFiles(file);
    } else {
      result.push({
        path: file,
        check: false,
        content: fs.readFileSync(file)
      });
    }
  });
}
```

result 保存了递归遍历的所有文件的 path、check、content 信息，其中 content 信息会被传给`prettier.js`的`check(content: string, options: object)`方法。

显然，上述代码是有错误的，但是极难发现。只有运行它的时候，才能通过堆栈报错来进行定位。**但如果借助 ts，就可以立即发现错误，保持代码稳健**。

这个问题放在文章最后再说，下面看看 ts 在 koa 项目中的运用吧。

## 项目目录

由于没有历史包袱，整个项目的架构还是非常清爽的。如下所示：

```bash
.
├── README.md
├── bin # 存放scripts的脚本文件
├── dist # 编译打包后的js文件
├── docs # 详细文档
├── package.json # npm
├── sh # pm2等脚本
├── src # 项目源码
├── tmp # 存放临时文件的地方
└── tsconfig.json # typescript编译配置
```

## typescript 编译与 npm 配置

因为是用 ts 来编写代码，因此需要专门编写 typescript 的配置文件：`tsconfig.json`。根据个人习惯，以及之前组内的 ts 项目，配置如下：

```json
{
  "compilerOptions": {
    "module": "commonjs", // 编译生成的模块系统代码
    "target": "es2017", // 指定ecmascript的目标版本
    "noImplicitAny": true, // 禁止隐式any类型
    "outDir": "./dist",
    "sourceMap": false,
    "allowJs": false, // 是否允许出现js
    "newLine": "LF"
  },
  "include": ["src/**/*"]
}
```

对于一些有历史遗留的项目，或者说用 js 逐步重构为 ts 的项目来说，由于存在大量的 js 遗留代码，因此`allowJs`这里应该为`true`，`noImplicitAny`应该为`false`。

在`package.json`中，配置两个脚本，一个是 dev 模式，另一个是 prod 模式：

```json
{
  "scripts": {
    "dev": "tsc --watch & export NODE_ENV=development && node bin/dev.js -t dist/ -e dist/app.js",
    "build": "rm -rf dist/* && tsc"
  }
}
```

在 dev 模式下，需要 tsc 监听配置中`include`中指定的 ts 文件的变化，并且实时编译。`bin/dev.js`是根据项目需要编写的监听脚本，它会监听`dist/`目录中编译后的 js 文件，一旦有满足重启条件，就重启服务器。

## 类型声明文件

koajs 与常见插件的类型声明都要在@types 下安装：

```sh
npm i --save-dev @types/koa @types/koa-router @types/koa2-cors @types/koa-bodyparser
```

## 区分 dev/prod 环境

为了方便之后的开发和上线，`src/config/`目录如下：

```typescript
.
├── dev.ts
├── index.ts
└── prod.ts
```

配置分为 prod 和 dev 两份。dev 模式下，向控制台打印信息；在 prod 下，需要向指定位置写入日志信息。类似的，dev 下不需要进行身份验证，prod 下需要内网身份验证。因此，利用 ts 的`extends`特性来复用数据声明：

```typescript
// mode: dev
export interface ConfigScheme {
  // 监听端口
  port: number;
  // mongodb配置
  mongodb: {
    host: string;
    port: number;
    db: string;
  };
}
// mode: prod
export interface ProdConfigScheme extends ConfigScheme {
  // 日志存储位置
  logRoot: string;
}
```

在 index.ts 中，通过`process.env.NODE_ENV`变量值来判断模式，进而导出对应的配置。

```typescript
import { devConf } from "./dev";
import { prodConf } from "./prod";

const config = process.env.NODE_ENV === "development" ? devConf : prodConf;

export default config;
```

如此，外界直接引入即可。但在开发过程中，例如身份认证中间件。虽然 dev 模式下不会开启，但编写它的时候，引入的`config`类型是`ConfigScheme`，在访问`ProdConfigScheme`上的字段时候 ts 编译器会报错。

这时候，ts 的断言就派上用场了：

```typescript
import config, { ProdConfigScheme } from "./../config/";

const { logRoot } = config as ProdConfigScheme;
```

## 中间件编写

对于整体项目，和 koa 关联较大的业务逻辑主要体现在中间件。这里以运营系统必有的「操作留存中间件」的编写为例，展示如何在 ts 中编写中间件的业务逻辑和数据逻辑。

引入 koa 以及编写好的轮子：

```typescript
import * as Koa from "koa";
import { print } from "./../helpers/log";
import config from "./../config/";
import { getDB } from "./../database/mongodb";

const { mongodb: mongoConf } = config; // mongo配置
const collectionName = "logs"; // 集合名称
```

操作留存中需要留存的数据字段有：

```
staffName: 操作人
visitTime: 操作时间
url: 接口地址
params: 前端传来的所有参数
```

ts 中借助 interface 直接约束字段类型即可。一目了然，对于之后的维护者来说，基本不需要借助文档，即可理解我们要和 db 交互的数据结构。

```typescript
interface LogScheme {
  staffName: string;
  visitTime: string;
  url: string;
  params?: any;
}
```

最后，编写中间件函数逻辑，参数需要指明类型。当然，直接指明参数是 any 类型也可以，但这样和 js 就没差别，而且也体会不到 ts 带来文档化编程的好处。

因为之前已经安装了`@types/koa`，因此这里不需要我们手动编写 `.d.ts` 文件。并且，koa 的内置数据类型已经被挂在了前面 import 进来的`Koa`上了（是的，ts 帮我们做了很多事情）。上下文的类型就是 `Koa.BaseContext`，回调函数类型是`() => Promise<any>`

```typescript
async function logger(ctx: Koa.BaseContext, next: () => Promise<any>) {
  const db = await getDB(mongoConf.db); // 从db链接池中获取链接实例
  if (!db) {
    ctx.body = "mongodb errror at controllers/logger";
    ctx.status = 500;
    return;
  }

  const doc: LogScheme = {
    staffName: ctx.headers["staffname"] || "unknown",
    visitTime: Date.now().toString(10),
    url: ctx.url,
    params: ctx.request.body
  };

  // 不需要await等待这段逻辑执行完毕
  db.collection(collectionName)
    .insertOne(doc)
    .catch(error =>
      print(`fail to log info to mongo: ${error.message}`, "error")
    );

  return next();
}

export default logger;
```

## 单元函数

这里以一个日志输出的单元函数为例，说一下「索引签名」的应用。

首先，通过联合类型约束了日志级别：

```typescript
type LogLevel = "log" | "info" | "warning" | "error" | "success";
```

此时，打算准备一个映射：日志等级 => 文件名称 的数据结构，例如 info 级别的日志对应输出的文件就是 `info.log`。显然，这个 object 的所有 key，必须符合 LogLevel。写法如下：

```typescript
const localLogFile: {
  [level in LogLevel]: string | void;
} = {
  log: "info.log",
  info: "info.log",
  warning: "warning.log",
  error: "error.log",
  success: "success.log"
};
```

如果对于 log 级别的日志，不需要输出到文件仅仅需要打印到控制台。那么`localLogFile`应该没有`log`字段，如果直接去掉`log`字段，ts 编译器报错如下：

```
Property 'log' is missing in type '{ info: string; warning: string; error: string; success: string; }' but required in type '{ log: string | void; info: string | void; warning: string | void; error: string | void; success: string | void; }'.
```

根据错误，这里将索引签名字段设置为「可选」即可：

```typescript
const localLogFile: {
  [level in LogLevel]?: string | void;
} = {
  info: "info.log",
  warning: "warning.log",
  error: "error.log",
  success: "success.log"
};
```

## 关于 export

**使用`export`导出复杂对象时候，请加上类型声明，不要依赖与 ts 的类型推断**。

`index.ts`：

```typescript
import level0 from "./level0";

export interface ApiScheme {
  method: ApiMethod;
  host: string;
}

export interface ApiSet {
  [propName: string]: ApiScheme;
}

export const apis: ApiSet = {
  ...level0
};
```

`level0.ts`:

```typescript
import { ApiSet } from "./index";

// 声明导出对象的数据类型
export const level0: ApiSet = {
  "qcloud.tcb.getPackageInfo": {
    method: "post",
    host: tcb.dataUrl
  },

  "qcloud.tcb.getAlarmRecord": {
    method: "post",
    host: tcb.dataUrl
  }
};
```

## 回到开头

回到开头的场景，如果用 typescript，我们会先声明`result`中每个对象的格式：

```typescript
interface FileInfo {
  path: string;
  check: boolean;
  content: string;
}

const result: FileInfo[] = [];
```

此时，你会发现 typescript 编译器已经给出了报错，在 `content: fs.readFileSync(file)` 这一行中，报错信息如下：

```bash
不能将类型“Buffer”分配给类型“string”。
```

如此，在编写代码的时候，就能立即发现错误。而不是写了几百行，然后跑起来后，根据堆栈报错一行行去定位问题。

仔细想一下，如果是 30 个人合作的大型 node/前端项目，出错的风险会有多高？定位错误成本会有多高？所以，只想说 ts 真香！

## 参考书籍

- [《TypeScript 入门教程》](https://ts.xcatliu.com/)
- [《TypeScript Deep Div》](https://basarat.gitbooks.io/typescript/content/docs/getting-started.html)
