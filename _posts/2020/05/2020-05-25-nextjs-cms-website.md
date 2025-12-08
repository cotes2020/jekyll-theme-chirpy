---
title: "基于 Next.js 和云开发 CMS 的内容型网站应用实战开发"
date: 2020-05-25
permalink: /2020-05-25-nextjs-cms-website/
categories: ["实战分享"]
---
## 引言


随着腾讯云云开发能力的日渐完善，有经验的工程师已经可以独立完成一个产品的开发和上线。**但网上云开发相关的实战文章非常少，很多开发者清楚云开发的能力，但是不清楚如何在现有的开发体系下引入云开发**。


![007S8ZIlgy1gez0z9axp8j31e10u0b2a.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gez0z9axp8j31e10u0b2a.jpg)


本文从云开发团队开发者+能力使用者的角度，以云开发官网 ([http://cloudbase.net/](http://cloudbase.net/)) 的搭建思路为例，分享云开发结合流行框架与工具的实战经验。


**涉及到的知识点有**：

- 云开发：
	- 扩展能力（CMS 扩展）
	- 静态托管
	- 云函数
	- 云数据库
	- CloudBase CLI 工具
- React 框架：Next.js
- CI 自动构建

## 背景介绍


随着云开发团队业务的迅猛发展，团队需要一个官网来更直观、更即时地向开发者们展示云开发的相关能力，包括但不限于工具链、SDK、技术文档等。


同时，为了降低开发者的上手成本，积累业界的优秀实战经验，官网也承载着营造社区氛围、聚合重要资料、增强用户黏度的重要任务。


我们最初使用 VuePress 作为静态网站工具，遇到了一些痛点：

- **问题 1: 每次更新内容，都需要配合 git。运营同学对 git 不熟悉**
- **问题 2: 学习资料方面的内容更新过于频繁，“污染”了 git 记录**
- **问题 3: 内容和网站代码耦合**
- **问题 4: 缺少可视化的内容编辑工具**

我们使用「CMS 扩展」、「云开发基础能力」、「Next.js」、「CI 工具」，很好地解决了以上问题。在实现网站内容动态化的同时，保证了 SEO，运营同学也可以通过 CMS 对内容进行可视化管理。


## 安装 CMS


进入[云开发扩展能力控制台](https://console.cloud.tencent.com/tcb/add)，根据引导，安装 CMS


内容管理系统。


![007S8ZIlgy1gezbg60ioij31fz0u0ncc.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbg60ioij31fz0u0ncc.jpg)


在最后进行扩展程序配置的时候，有两种账号：管理员账号和运营者账号。管理员账号权限更高，可以创建新的数据集合；而运营者账号只能在已有的数据集合上进行增删改的操作。


![007S8ZIlgy1gezbi2twy0j31gl0u041d.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbi2twy0j31gl0u041d.jpg)


> 注意：  
> 安装时间有些长，请耐心等待


安装成功后，云数据库会自动创建 3 个集合：`tcb-ext-cms-contents`、`tcb-ext-cms-users`、`tcb-ext-cms-webhooks`。


![007S8ZIlgy1gezbngm3y6j31nt0u0q8f.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbngm3y6j31nt0u0q8f.jpg)


会自动创建 3 个云函数：`tcb-ext-cms-api`、`tcb-ext-cms-init`、`tcb-ext-cms-auth`。


![007S8ZIlgy1gezbpx8nrhj31lh0u0tes.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbpx8nrhj31lh0u0tes.jpg)


进入「静态网站托管」，可以看到 CMS 系统的静态文件已经自动上传到`tcb-cms/`目录下了：


![007S8ZIlgy1gezbunf926j31mt0u0afk.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbunf926j31mt0u0afk.jpg)


点击上方的「基础配置」，就可以查看到域名信息。


![007S8ZIlgy1gezbvpy0fpj31a70u0dnt.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbvpy0fpj31a70u0dnt.jpg)


在浏览器中访问：[http://pagecounter-d27cfe-1255463368.tcloudbaseapp.com/tcb-cms/](http://pagecounter-d27cfe-1255463368.tcloudbaseapp.com/tcb-cms/) 即可看到 CMS 系统：


![007S8ZIlgy1gezbxipje4j31oi0tokge.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezbxipje4j31oi0tokge.jpg)


到此为止，无任何开发成本，一个 CMS 内容管理系统就正式上线了～


## 使用 CMS 创建动态内容


对于动态化的数据内容，我们将其划分为不同的模块。每个内容模块，对应 CMS 系统的一个数据集合。例如「云开发官网」-「社区页」中，推荐好课的内容就是动态的。


![007S8ZIlgy1gezc29itjnj31e10u0kjl.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezc29itjnj31e10u0kjl.jpg)


从图中可以看到，每节课程有着多个属性。而在云数据库中，每节课程就对应一个文档，课程属性就对应文档的字段。字段类型与含义如下：


```typescript
name<string>: 课程名称
time<number>: 课程时间
cover<string>: 课程封面
url<string>: 课程链接
level<0 | 1 | 2>: 课程难度
```


以管理员身份登录 CMS 系统，在「内容设置页」新建内容。在 CMS 中，支持多种高级数据类型，例如 url、图片、markdown、富文本、标签数组、邮箱、网址等，并对这些类型进行了智能识别和更友好地展示。


> 注意：  
> CMS 自带图床功能。当数据类型是「图片」时，图片会自动上传到当前云开发环境下的云存储中。图片信息以 cloud:// 开头的特殊链接，存放在数据集合中。


新建内容时，默认情况下，CMS 会自动填充 4 个字段：name、order、createTime、updateTime。可以根据自身需要，对不需要的字段进行删除。


> 建议：  
> 保留 order 字段，它可以被用作数据排序。对运营者来说，数据的 order 的值越大，在 CMS 系统中展示的位置越靠前；对开发者来说，可以根据 order 来进行排序搜索。从而保证了体验和逻辑的一致性。


根据字段创建集合后，CMS 系统左侧会看到「推荐好课」。它对应的内容被保存在云数据库的`recommend-course`（创建时指定）集合中，它的字段信息保存在云数据库的`tcb-ext-cms-contents`（CMS 初始化时创建）集合中。


![007S8ZIlgy1gezcvlw9b4j312b0u0gu1.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezcvlw9b4j312b0u0gu1.jpg)


按照设定添加新的课程内容后，再次进入「推荐好课」，如下所示：


![007S8ZIlgy1gezcwwgnb0j322c0tk44l.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezcwwgnb0j322c0tk44l.jpg)


图片、链接等内容，更友好地展示给运营者。


## 项目搭建


按照 [Next.js 文档](https://nextjs.frontendx.cn/) 的指引，创建 Next.js 项目：


```shell
npm i --save next react react-dom axios
```


因为我们要将网站部署到「静态托管」上，所以要使用 Next.js 的**静态导出**功能。`package.json` 中的打包脚本更新为：


```json
"scripts": {
    "dev": "next",
    "build": "next build && next export",
    "start": "next start"
}
```


为了快速部署静态网站，以及发布云函数。需要全局安装 `@cloudbase/cli`:


```shell
npm install -g @cloudbase/cli
```


安装后，添加两个脚本：

- `deploy:hosting`: 将 Next.js 的静态导出文件部署到「静态托管」
- `deploy:function`: 发布项目中的云函数

```json
"scripts": {
    "deploy:hosting": "npm run build && cloudbase hosting:deploy out -e jhgjj-0ae4a1",
    "deploy:function": "echo y | cloudbase functions:deploy --force"
}
```


> 注意：  
> 准备两个云环境，防止静态部署时文件覆盖。envId 为 jhgjj-0ae4a1 的云环境只用于部署 Next.js 的静态导出文件。envId 为 pagecounter-d27cfe 的云环境用来部署 CMS 系统。


## 获取 CMS 内容


### 编写云函数


为了能在 Next.js 中读取到 CMS 系统的最新数据，我们需要新建一个云函数，配合 HTTP Service，解析 Next.js 传入的参数，读取云数据库中的信息，并且返回给 Next.js。


> 为什么需要云函数配合 HTTP Service，不能直接使用 SDK 吗？  
> Next.js 预渲染的环境不支持 SDK（tcb-admin-node、tcb-js-sdk）。在 Next.js 中，动态获取数据注入到模板变量时，需要在 getInitialProps() 方法中进行异步操作。我们使用 axios（支持 ssr 环境），通过访问 url（云开发 HTTP Service 能力）触发云函数，获取最新数据。


在项目目录下创建`config.js`，存放一些配置信息：


```javascript
module.exports = {
    envId: "pagecounter-d27cfe", // 云开发环境envid
    siteAuthKey: "QhBYWnRjijGcGTBUxDFGWxuq", // 用于site-cms-data云函数中的身份校验
    httpPath: "/site-cms-data" // site-cms-data云函数的http触发路径
};
```


创建 CloudBase CLI 工具的配置文件`cloudbase.js`，用于云函数部署:


```javascript
const { envId, siteAuthKey } = require("./config");
module.exports = {
    envId,
    functionRoot: "./cloudfunctions",
    functions: [
        {
            name: "site-cms-data",
            config: {
                // 超时时间
                timeout: 30,
                // 环境变量
                envVariables: {
                    SITE_AUTH_KEY: siteAuthKey
                },
                runtime: "Nodejs8.9",
                installDependency: true
            },
            handler: "index.main"
        }
    ]
};
```


创建 `cloudfunctions/site-cms-data/` 目录，里面存放云函数的主要逻辑。


![007S8ZIlgy1gezw041pcrj30k807i0tf.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezw041pcrj30k807i0tf.jpg)


`provider.js`中提供 Provider 对象，它支持：

- `fetchAll()`：获取指定集合的所有数据
- `query()`：支持`orderBy`、`where`的条件查询

之后我们会在 Next.js 端传入名为`api`的参数，它必须是 Provider 支持的方法。`dispatch()`会根据前端传入的`api`，自动调用 Provider 上的方法。


```javascript
const Provider = {
    // 获取指定集合的所有数据
    async fetchAll(ctx) {
        const {
            db,
            params: { collectionName }
        } = ctx;
        const collection = db.collection(collectionName);
        return collection
            .where({
                _id: /.*/
            })
            .get();
    },
    // 支持orderBy、where的条件查询（满足当前需求）
    async query(ctx) {
        const {
            db,
            params: { collectionName, orderBy, where }
        } = ctx;
        let promise = db.collection(collectionName);
        if (Array.isArray(orderBy) && orderBy.length === 2) {
            promise = promise.orderBy(...orderBy);
        }
        if (typeof where === "object") {
            promise = promise.where(where);
        }
        return promise.get();
    }
};
async function dispatch(ctx) {
    return await Provider[ctx.api](ctx);
}
module.exports = {
    Provider,
    dispatch
};
```


`interceptor.js`提供：

- `vertifyAuth()`: 用户身份检验
- `isValidBody()`: 参数类型检验

```javascript
const { Provider } = require("./provider");
const supportedApi = Reflect.ownKeys(Provider);
function vertifyAuth(ctx) {
    // 前面在cloudbase.js中规定的环境变量
    return ctx.key === process.env.SITE_AUTH_KEY;
}
function isValidBody(body) {
    return (
        "key" in body && // 验证身份的随机密钥
        "params" in body && // 携带调用服务需要的参数
        "api" in body && // 调用服务名称
        supportedApi.includes(body.api)
    );
}
module.exports = {
    vertifyAuth,
    isValidBody
};
```


在 `index.js` 中，封装了云函数的整体逻辑：

1. 检验 Next.js 端传入数据是否合法
2. 检验身份密钥，防止云数据库被盗刷
3. 绑定特殊变量到上下文，减少 tcb 对象的实例化次数
4. 调用对应服务，返回结果

```javascript
const tcb = require("tcb-admin-node");
const { vertifyAuth, isValidBody } = require("./interceptor");
const { dispatch } = require("./provider");
module.exports.main = async (event, context) => {
    let ctx = {
        envId: context.namespace
    };
    // 验证传入的数据
    try {
        const body = JSON.parse(event.body);
        if (isValidBody(body)) {
            ctx = {
                ...ctx,
                ...body
            };
        } else {
            return {
                success: false,
                msg: "传入数据不合法"
            };
        }
    } catch (error) {
        console.error(error);
        return {
            success: false,
            msg: "请检查body格式"
        };
    }
    // 验证身份
    if (!vertifyAuth(ctx)) {
        return {
            success: false,
            msg: "身份验证失败"
        };
    }
    // 给上下文绑定db
    const app = tcb.init({
        env: ctx.envId
    });
    ctx.db = app.database({
        env: ctx.envId
    });
    // 服务调用
    try {
        const result = await dispatch(ctx);
        return {
            success: true,
            result
        };
    } catch (error) {
        console.error(error);
        return {
            success: false,
            msg: error.message
        };
    }
};
```


> 建议：  
> 在实际开发过程中，请规范云函数的结果返回格式。以此云函数为例，成功时返回: {success: true, result: 结果}，失败时返回: {success: false, msg: 错误信息}


### 发布云函数


通过 CloudBase CLI 工具的命令，清空之前的登录状态，重新进行登录：


```shell
cloudbase logout && cloudbase login
```


在项目目录下，执行发布云函数的命令：


```shell
npm run deploy:function
```


在「云开发控制台」-「云函数页」中，可以看到云函数`site-cms-data`上传成功：


![007S8ZIlgy1gezzkmla7ej327g0u0wlx.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezzkmla7ej327g0u0wlx.jpg)


进入云函数`site-cms-data`，在「函数配置」中，修改“HTTP 触发路径”（和 `config.js` 中的 `httpPath` 字段保持一致）：


![007S8ZIlgy1gezznghm7lj30xt0u0n3o.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gezznghm7lj30xt0u0n3o.jpg)


成功后，我们可以就可以通过 `https://${envId}.service.tcloudbase.com/site-cms-data` 来触发此函数。


### 在 Next.js 中获取动态数据


在云函数 `site-cms-data` 中，只解析外界传入的 3 个参数：


child_database


为了避免每次都重复编写 axios 的请求配置，将一些共用的信息抽离出来, `generateAxiosConfig()` 实现如下:


```typescript
// provider.js
import { siteAuthKey } from "./config";
/**
 * 为axios生成请求配置
 * @param {String} api 云函数(site-cms-data)支持的服务
 * @param {Object} params 服务入参
 */
function generateAxiosConfig(api, params) {
    const data = {
        key: siteAuthKey,
        api,
        params
    };
    const config = {
        headers: {
            "Content-Type": "application/json"
        },
        method: "POST",
        data: JSON.stringify(data)
    };
    return config;
}
```


前文有讲到，CMS 自带图床功能，拖拽上传的图片会被存储在同一环境下的云存储中，并且获取图片的链接存放在集合中。**云存储的链接是以 \**_**\**_`cloud://`\_**\**_ **开头的特殊链接，需要在前端进行识别和特殊处理**。


以前文我们上传的图片为例，它的链接是：`cloud://pagecounter-d27cfe.7061-pagecounter-d27cfe-1255463368/uploads/1589990230404.png`。将其转成可访问的 http 链接：`https://7061-pagecounter-d27cfe-1255463368.tcb.qcloud.la/uploads/1589990230404.png`。


转换思路是：识别 envid 后的信息，将其与`tcb.qcloud.la`域名重新拼接即可。代码实现如下：


```typescript
// provider.js
/**
 * 获取云存储的访问链接
 * @param {String} url 云存储的特定url
 */
function getBucketUrl(url) {
    if (!url.startsWith("cloud://")) {
        return url;
    }
    const re = /cloud:\\/\\/.*?\\.(.*?)\\/(.*)/;
    const result = re.exec(url);
    return `https://${result[1]}.tcb.qcloud.la/${result[2]}`;
}
```


> 注意：  
> 云存储的「权限设置」应为：所有用户可读，仅创建者及管理员可写。否则链接无法访问。


> 推荐：  
> 除了自带的图床功能，开发者可以根据自身需求使用其他稳定图床服务，例如微博图床。如果使用其他图床，对应字段类型不能设置为「图片」，可以是「字符串」或者「超链接」。


在 `provider.js` 中对外暴露 `getCourses()` 方法，获取「推荐课程」的数据，并且进行处理：


```typescript
import { siteAuthKey, envId, httpPath } from "./config";
import axios from "axios";
const url = `http://${envId}.service.tcloudbase.com${httpPath}`;
/**
 * 获取推荐课程数据
 */
export async function getCourses() {
    const config = generateAxiosConfig("fetchAll", {
        collectionName: "recommend-course"
    });
    const res = await axios(url, config);
    const { success, result, msg } = await res.data;
    if (success) {
        return result.data.map(item => ({
            ...item,
            cover: getBucketUrl(item.cover) // 处理云存储的特殊链接
        }));
    } else {
        throw new Error("获取「推荐课程」失败:" + msg);
    }
}
```


创建 `pages/index.js` 文件，它对外暴露的函数组件`HomePage`(被渲染为首页)。我们在组件上的`getInitialProps()`方法中获取推荐课程数据，并且将其注入到组件的`props`上：


```text
import React, { useState } from "react";
import { getCourses } from "./../provider";
const HomePage = props => {
    // ...
};
HomePage.getInitialProps = async () => {
    const promises = [getCourses()];
    const [courses] = await Promise.all(promises);
    // 返回组件的props
    return { courses };
};
export default HomePage;

```


在 HomePage 中，可以从 `props` 中读取到推荐课程数据，将其渲染到页面上即可：


```typescript
const levelMap = {
    0: "初级",
    1: "中级",
    2: "高级"
};
const HomePage = ({ courses }) => {
    return (
        <>
            {courses.map((course, index) => (
                <div key={index}>
                    <p>
                        <a href={course.url}>{">>> 立即学习"}</a>
                    </p>
                    <p>
                        <strong>课程名称：</strong> {course.name}
                    </p>
                    <p>
                        <strong>课程时长：</strong> {course.time} 课时
                    </p>
                    <p>
                        <strong>课程难度：</strong>
                        {levelMap[course.level]}
                    </p>
                    <p>
                        <strong>课程封面：</strong>
                        <img src={course.cover} />
                    </p>
                </div>
            ))}
        </>
    );
};
```


打开浏览器，进入 [http://localhost:3000/](http://localhost:3000/)


，可以看到效果如下：


![007S8ZIlgy1gf006o852jj319d0u0guf.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gf006o852jj319d0u0guf.jpg)


进入 view-source:[http://localhost:3000/](http://localhost:3000/) ，可以看到网页的 html 源码中包含了课程数据，解决了 SEO 的问题：


![007S8ZIlgy1gf007i7i7hj32la0goaq2.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gf007i7i7hj32la0goaq2.jpg)


> 注意： > getInitialProps()方法会将数据序列化，它执行于编译时期，而不是在网页生命周期中触发的。


## 自动构建与部署


目前为止，开发工作基本结束。执行 `npm run build` 命令，网站静态文件被打包到了 `out/` 目录下：


![007S8ZIlgy1gf00fkl5j1j30m80ucdj0.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gf00fkl5j1j30m80ucdj0.jpg)


执行 `npm run deploy:hosting` 将 `out/` 目录下的文件上传到「静态网站托管」。访问静态网站托管的链接：[https://jhgjj-0ae4a1.tcloudbaseapp.com/](https://jhgjj-0ae4a1.tcloudbaseapp.com/) ，效果如下：


![007S8ZIlgy1gf00jcjxjsj31o90u0wsy.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-05-25-nextjs-cms-website/007S8ZIlgy1gf00jcjxjsj31o90u0wsy.jpg)


借助成熟的 CI 工具，例如 Travis、Circle 等，可以定时触发构建工作。如此一来，内容和开发彻底分离。


在构建发布的时候，需要用到 CloudBase CLI 工具。在 CI 工具中，不再使用 `cloudbase login` 进行交互式输入登录，而是使用密钥登录： `cloudbase login --apiKeyId $TCB_SECRET_ID --apiKey $TCB_SECRET_KEY` 。


> 注意：  
> 前往 云 API 密钥 获得 TCB_SECRET_ID 和 TCB_SECRET_KEY 的值


在 CI 工具的控制台中，配置 `TCB_SECRET_ID` 和 `TCB_SECRET_KEY`。并为`package.json`新添加一个脚本：


```json
"scripts": {
    "login": "echo N | cloudbase login --apiKeyId $TCB_SECRET_ID --apiKey $TCB_SECRET_KEY"
}
```


总结来说，CI 构建的流程是：

- tcb 密钥登录：`npm run login`
- 获取最新数据，导出静态文件：`npm run build`
- 发布到「静态网站托管」：`npm run deploy:function`

如果数据需要紧急修改上线，可以在本地或者 CI 工具控制台，手动触发构建。


## 最后


在现有开发体系下，合理运用云开发，使得人力成本、开发成本以及运维成本大幅度降低。前后端一把梭，构成“闭环”。


本文实战仅是抛砖引玉，涉及了云开发能力的一部分，还有更多好玩的东西等待你的探索，比如使用云函数实现 SSR、托管后端服务、图像服务、各端 SDK 等。


**探索能力，发散思路，以更低成本开发高可用的 web 服务，云开发绝对是你最好的选择！**


**更多资料**：

- [云开发社区官网](https://cloudbase.net/)
- [CMS 内容管理系统](https://docs.cloudbase.net/extension/abilities/cms.html)

