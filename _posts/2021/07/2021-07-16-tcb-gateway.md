---
title: "十亿级日流量--微信云开发网关架构设计"
date: 2021-07-16
permalink: /2021-07-16-tcb-gateway/
categories: ["C工作实践分享"]
---
## 背景


本次的主题是TCB云开发网关的架构设计，分享业务型网关的技术选型、功能设计、优化演进等方面的经验。目前云开发网关的每日请求量20亿+，压测时可承受百亿峰值流量，系统整体QPS平均在 45325。


在阅读之前，需要准备具备或者了解一些前置知识：

- 使用/了解过 Serverless
- 开发/了解过 网关 or BFF
- 开发/了解过 NodeJS and （Express or Koa or NestJS）
- 了解过 Docker and K8S and Redis and ELK and Unix

## 认识网关


### 分层架构


> “计算机科学领域的任何问题都可以通过增加一个间接的中间层来解决。”  
> — David Wheeler


说到网关，肯定要提到「分层架构」。分层架构是软件设计中最常见的一种架构方式，比如TCP/IP体系、操作系统体系。


![008i3skNgy1gsiqp137yfj307k0fyq31.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqp137yfj307k0fyq31.jpg)


分层架构的核心是保证隔层之间边界明显，从而保证逻辑高内聚；层与层之间依赖于事先定好的约定，只能按照某个方向单向进行调用；并可以通过加层，快速扩展新的逻辑。


### 三层C/S 架构


随着多端设备的兴起，多端开发逐渐成为主流。虽然处于不同端，但相同业务，都会使用同一份数据。当不同端的请求走到了后台，后台去数据库查询数据，并且将数据拼接返回给前端。


![008i3skNgy1gsiqq9iozqj30w00p7myv.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqq9iozqj30w00p7myv.jpg)


### 四层 C/S 架构


为了保证服务可用性、接口性能、数据安全，后端开发往往需要考虑缓存、节流、降级、鉴权等功能。这些功能并不和某个特定业务强关联，并且在各个服务中都是通用的。


> 按照分层架构的思想，将这些功能放在单独的一层中，称为网关；提供功能的服务，称之为网关服务。


![008i3skNgy1gsiqv2p4g4j31f00u0tbk.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqv2p4g4j31f00u0tbk.jpg)


## 技术选型


### 开发语言


child_database


云开发网关的开发语言是NodeJS。为什么选择NodeJS？

- 异步非阻塞的编程模型，`async/await`语法让开发者用同步的写法写出异步的代码
- 本身适合IO密集型场景，不需要太多前置知识就能写出性能较高的代码
- TypeScript的推广，使得类型系统被应用在大型JS项目中
- 团队在云开发数据流服务（百亿流量）、腾讯云中间层服务等Node服务中，积累了丰富的经验和相关工具

> NodeJS是最优选择吗？


实际上，Node也有很多缺陷。举几个例子：

- 单线程的瓶颈限制
- NodeJS进行数据计算性能非常低
- 相较于Java，官方没有支持高级的数据结构和算法

**不过，大多数场景下，完全触及不到语言的瓶颈，后面也有其它方法来提高吞吐。同时脱离场景，用简单的demo 单纯地比对语言的性能，是没有任何实际意义的。**


### 开发框架


child_database


在对比了社区中常用的NodeJS框架之后，结合团队内部的实践经验，最终选择了NestJS作为开发框架。


## 应用场景


云开发网关是业务型网关，主要的应用场景是Web端开发，为底层的云函数（Scf）、云托管（Docker）提供HTTP访问形式。


> 新功能  
> 目前也支持静态网站托管中的静态资源（COS）


![008i3skNgy1gsiqul6f46j31s60u078z.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqul6f46j31s60u078z.jpg)


如图所示，通过`默认域名/触发路径`访问，就能触发云函数执行并且得到返回结果，或者得到经过CDN缓存的静态网站托管中的资源。


以图中的lcap-business-service云函数为例，访问[https://lowcode-9gqz7bmk009337cd-1301116672.ap-shanghai.app.tcloudbase.com/lcap-business-service](https://lowcode-9gqz7bmk009337cd-1301116672.ap-shanghai.app.tcloudbase.com/lcap-business-service)就能触发云函数的执行，并且返回结果：


```json
{
    "code": 100000,
    "result": "Hello Lowcode Business Service",
    "reqId": "185d75e2-ad4b-4fa2-a076-1288286826c0",
    "scfReqId": "e5817cc8-e3d8-11eb-9c97-525400ff21a1"
}
```


## 整体请求流程


![008i3skNgy1gsiqt6v5w0j32ek0quq5t.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqt6v5w0j32ek0quq5t.jpg)


前端访问默认域名`xxxxxx.ap-shanghai.app.tcloudbase.com`，会被路由到k8s网关集群上。


网关服务根据触发路径的类型（SCF or Docker），将不同类型的流量转发到不同的底层服务上。


## 代码模块设计


基于NestJS提供的概念，将代码模块划分为4大模块：

- 控制器（Controller）：处理路由逻辑，负责将接收到的请求转发给底层服务
- 过滤器（Filter）：使用AOP的方式，捕获全局错误，并且统一返回
- 中间件（Middleware）：对请求进行修改，比如处理跨域、生成完整的请求上下文信息，并将请求传递给controller
- 提供者（Provider）：封装各种功能提供给controller调用，比如dns解析、配置读取、异步请求等

![008i3skNgy1gsiqvqqdc9j31ry0rcadu.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqvqqdc9j31ry0rcadu.jpg)


在这4大模块中，Provider的功能最为复杂。为了让代码可扩展性更高，参考分层架构设计，将Provider整体划分成了3层。


child_database


并且只能高级Layer中的模块，调用低级Layer中的模块，不能反向调用，从而避免循环引用的产生。


> **为什么要避免循环引用？**  
> NestJS在启动项目时，需要解析并且实例化各个Provider，并将其进行依赖注入。如果两个Provider之间存在循环引用，那么就会导致解析失败。  
> NestJS针对循环引用提供了解决方案，但是在实践中出现了各种奇怪问题，排查难度大。因此要从代码设计上尽量规避循环引用的产生。


> **在实际编码中，如何避免循环引用？**  
> 假设目前有providerA和providerB，并且providerA要调用providerB的methodB()方法，providerB要调用providerA的methodA()方法。  
> 为了避免循环引用，定义providerC，并且将methodA()和methodB()放入providerC中。从而providerA和providerB都引用providerC，但不会互相引用。


## 配置模块


> 配置的作用是什么？  
> 能够让开发者在开发时，快速连通不同的环境，从而实现快速测试、联调、上线。


常见的配置有2种类型：

- 命令行变量：在NodeJS中，可以通过`process.env`读取到命令行变量，一般将其放在`.env`文件中。
- 配置文件：一般将其放在`/config`文件夹下，需要根据不同的环境，编写不同的配置文件。

**在本地开发时，配置模块的处理流程是：**

1. 先读取项目目录下的`.env`文件，并且进行解析
2. 将解析结果放入到`process.env`上
3. 根据`.env`中的环境信息和地域信息，读取并加载对应的ts配置文件。例如对于上海地域的生产环境，配置文件名就是`config/sh.production.config.ts`

**线上运行时，配置模块的处理流程是：**


`.env`文件不会放入git版本库中，当服务部署到K8s上之后，环境变量是通过`ConfigMap`来配置的。
对于不同地域不同环境下的服务，将对应ConfigMap映射为服务Pod中的`.env`文件。当Pod启动时，处理流程和本地开发一样。


**一个简易版的代码实现：**


```typescript
import { Injectable } from '@nestjs/common';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import { LoggerService } from './logger.service';
// 命令行变量类型声明
const ENV_KEYS = {
    NODE_ENV: 'NODE_ENV',
    REGION: 'REGION'
} as const;
export type ENV_KEYS_TYPE = keyof typeof ENV_KEYS;
// 配置文件类型声明
interface ConfigSchema {
    qpsLimit: number // QPS 限制
}

@Injectable()
export class EnvService {
    private config: ConfigSchema
    constructor(private readonly loggerService: LoggerService) {
        this.loadEnv();
        this.loadConfig();
    }
    // 1、加载命令行变量
    private loadEnv() {
        const envPath = path.join(process.cwd(), '.env');
        const isExist = fs.existsSync(envPath);
        if (!isExist) {
            return this.loggerService.info({ logType: 'EnvLoadFail', content: `Please create ${envPath}` });
        }

        const { parsed } = dotenv.config({ path: envPath });
        const envs = <Record<ENV_KEYS_TYPE, string>>parsed;
        for (const key in envs) {
            process.env[key] = envs[key];
        }
    }
    // 2、读取配置文件
    private loadConfig() {
        const mode = this.getEnvironmentVariable('NODE_ENV')
        const region = this.getEnvironmentVariable('REGION')
        const filePath = path.resolve(process.cwd(), `${region}.${mode}.ts`)
        this.config = require(filePath) as ConfigSchema;
    }
    // 其它Provider通过这2个方法读取命令行变量和配置文件变量
    public getEnvironmentVariable(key: keyof typeof ENV_KEYS) {
        return process.env[key];
    }
    public getConfigVariable(key: keyof ConfigSchema) {
        return this.config[key]
    }
}
```


## 日志模块


> 日志的作用是什么？  
> 能够帮助开发者快速复原请求链路上的关键节点的信息，作为定位问题、优化系统性能、监控告警的依据。


### 日志结构设计


**设计一个日志模块，首先就要考虑到日志的结构**。日志结构的设计原则是：

- 字段名尽量简短，可以适当添加业务前缀，增强可读性
- 字段值需要有最大长度，防止日志过多撑爆ELK

一个比较完备的日志结构设计是：


```typescript
// 请求上下文信息
interface IRequestCtxBaseInfo {
  reqId?: string; // 中间层生成的 version4 的 uuid ，可以用于快速查询scf日志
  reqStartTime?: number; // 请求开始时间
  reqCostTime?: number; // 从接收到服务到打印日志时的耗时
  reqPath?: string;
  reqMethod?: string;
  reqBody?: string;
}
// 日志等级
type ILogLevel = 'info' | 'error' | 'warn';
// 日志结构
interface ILogInfo extends IRequestCtxBaseInfo {
  logType: string; // 当前日志记录的类型，可以用于在一次请求的多个日志记录中，快速定位查找所需信息
  logLevel?: ILogLevel; // 日志等级
  // 错误信息
  errStatus?: number;
  errMsg?: string;
  errStack?: string;
  // 缓存信息
  cacheType?: 'miss' | 'expire';
  cacheKey?: string;
  cacheVal?: string;
  // action 接口信息
  actionName?: string;
  actionParams?: string;
  actionRes?: string;
  // 其它信息
  content?: string;
}

```


> **什么是reqId？**  
> 任何请求都要生成一个针对请求的唯一标识，这就是reqId。  
> 分布式唯一ID的生成算法有数据库生成法、Snowflake算法、UUID算法等等，项目中使用的是UUID算法。


### 日志埋点


**其次，要考虑到打日志的时机**。一般会以下几个场景中打印日志：

- 中间件：比如在请求进来时，打印logType为`IncomingRequest`的日志信息。此日志可以用来统计流量。
- 提供者：比如在缓存模块中，读取redis缓存时，打印缓存的命中信息。此日志可以用来优化缓存的设计，提高缓存命中率。
- 业务逻辑：比如在触发云函数模块中，打印触发结果信息。此日志可以用来进行监控告警。
- 其他关键节点

### 日志存储和展示


**最后，要考虑日志的展示和存储。**


本地开发时，需要根据日志类型，高亮日志，效果如下（可以清晰的看出错误日志、普通日志、告警日志）：


![008i3skNgy1gsiqwgf0p1j31l80gwwlt.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqwgf0p1j31l80gwwlt.jpg)


对于线上服务，不用高亮日志，将日志统一上传到ELK，交由ELK来收集和展示。


本地开发时，不需要专门存储日志信息，直接将日志打印到标准输出流即可；


对于线上服务，需要进行「双写」，将日志写到Pod的标准输出流；同时将日志写入到指定文件中，此文件会被相关脚本收集进行上报。


## 监控告警模块


> 监控告警的作用是什么？  
> 监控告警能快速发现线上服务的问题，并且将问题下发给开发者，介入恢复服务。


### 监控告警链路


![008i3skNgy1gsiqwwayq3j31ok0l2gmy.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqwwayq3j31ok0l2gmy.jpg)


### DSL 告警规则设计


图中的监控服务是一个独立定时运行的任务服务。它按照配置的规则，定时从ELK中查询日志，当查询的日志满足配置规则中的告警字段设置时，发起告警。


配置规则是一套自定义的DSL，它的格式如下：


```yaml
version: v2

dataSource:
  es:
    index: gateway_log-*

rules:
  - title: 查询底层服务集群失败
    alarm:
      qcloud:
        type: GATEWAY-ALARM
    interval: 1m # 每隔1分钟查一次日志系统
    tags: TCB
    period:
      length: 1m
      count: 1
    monit: # 传给 ELK 的查询语句和参数
      type: Metric
      thresholds: [0, 10] # 合法阈值是10次以内，超过10次，发起告警
      args:
        filters:
          query: logType:describeClusterFailed AND kubernetes.owner_name:gateway
```


## 名字服务模块


### 认识名字服务


![008i3skNgy1gsiqxc5wh9j30s40ug0tq.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqxc5wh9j30s40ug0tq.jpg)


在网关内部，调用其他服务时，需要服务的地址，也就是`ip:port`。**在微服务的架构下，并不会在代码中直接写明****`ip:port`****，而是有一个专门的名字服务，用来做服务注册、服务发现等逻辑**。


> 为什么需要名字服务？ip:port不容易在代码中维护和管理。在名字服务中，使用的是服务名字。服务的名字通常是不变的，但是其对应的ip:port可以有很多个。服务提供方也可以在使用方无感知的情况下，快速扩展或者裁撤服务IP。


### 调用流程


例如要调用底层scf服务的地址，在没有名字服务前，调用伪代码是：


```typescript
async function callFunction() {
  await axios({
    url: '<http://8.234.34.123:5566>', // 服务地址是硬编码
    method: 'post',
    data: {
      // ...
    }
  })
}
```


有了名字服务之后，处理逻辑是：

- 引入对应的名字服务的SDK
- 传入底层服务的名字，查询底层服务的地址
	- 成功：返回底层服务地址
	- 失败：向名字服务上报错误，并且代码内抛出错误

伪代码是：


```typescript
const nameServerSdk = require('....') // 引入SDK
async function resolveServerIpPort(serverName: string) {
  const res = await nameServerSdk.find(serverName) // 查询服务地址
  if (res.err) {
    nameServer.report(res.err); // 异步上报查询失败错误
    throw new Error(res.err);
  }
  return `${res.result.ip}:${res.result.port}`
}
async function callFunction() {
  const url = await resolveServerIpPort('Production:SCF');
  await axios({
    url,
    method: 'post',
    data: {
      // ...
    }
  })
}
```


## DNS模块


### 系统瓶颈


![008i3skNgy1gsiqxqjwvoj32ay0u042g.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqxqjwvoj32ay0u042g.jpg)


在云开发中，支持配置自定义域名，解析到默认域名上。


注意这里默认域名的格式：`lowcode-9gqz7bmk009337cd-1301116672.ap-shanghai.app.tcloudbase.com`。域名中是带有一些信息的，例如云开发环境ID、腾讯云APPID。这些信息会被用作鉴权等逻辑。


当用户通过自定义域名访问时，网关服务里请求信息的域名是用户的自定义域名，而不是默认域名。此时，为了查到用户的信息，需要通过DNS查询得到默认域名，再解析默认域名。


问题出现了，通常DNS查询非常耗时，尤其是在本地没有DNS缓存的情况下。这里的几十毫秒的耗时，对于网关系统来说，非常致命。


> 一般来说，网关系统的转发耗时应该控制在10ms左右。


### DNS 服务设计


![008i3skNgy1gsiqxyxjd3j316f0u075m.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqxyxjd3j316f0u075m.jpg)


为了优化DNS查询耗时，参考HTTP DNS的设计，专门独立出一个DNS服务。它会定时（一般是600s）去扫描用户绑定的自定义域名，然后将最新的DNS解析结果存储到Mysql中，并且将记录同步到Redis中。


> 为什么需要一层Redis？  
> 通过读写分离，进一步提高系统的读性能，并且防止大批量写入操作阻塞读操作。


Gateway识别出访问的是自定义域名之后，会走内网链路，调用DNS服务，DNS服务会读取Redis中的记录，返回给Gateway。如果结果为空，那么Gateway兜底走本地的DNS解析。


> 为什么不能在用户设置/修改DNS解析时，保存解析记录呢？  
> 用户的域名服务不一定托管在同一云厂商，云开发服务无法感知用户对DNS的操作。


## 流量转发模块


![008i3skNgy1gsiqxc5wh9j30s40ug0tq.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqxc5wh9j30s40ug0tq.jpg)


在转发请求时，网关会在HTTP请求头部中添加一些metadata，比如云开发用户名、云开发环境等。


在和底层服务建立连接之后，有2种数据传输的方式：

- 分块传输：用于云开发SCF
- 流式传输：用于云托管和静态托管

### 分块传输


在分块传输中，网关先调用底层云函数的接口，拿到云函数的返回结果后，将其返回给前端。伪代码如下：


```typescript
async function scfHandler() {
  const scfRes = await axios({
    url: '<http://1.1.1.1:4345>',
    method: 'post',
    data: {
      // ...
    }
  })

  return scfRes
}
```


### 流式传输


![008i3skNgy1gsiqyv3ckvj311a086t8u.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsiqyv3ckvj311a086t8u.jpg)


在流式传输中，网关承载的角色更像是一个管道（pipe），传输的数据不再是一整份数据，而是流（stream）。


**为什么采用流式传输呢？**


流式传输能够更好的处理大数据，并且不会阻塞NodeJS，从而提高并发数。


> 为什么SCF不使用流式传输？  
> SCF底层服务只支持Chunked Transfer.


**在NodeJS是如何实现流式传输呢？**


这里使用的第三方库是`got.js`。在npm上，它的安装量已经远超`axios`。它和`axios`最大的区别是，它支持返回Stream流对象；并且它是专门在NodeJS中使用，没有`axios`的adapter，性能更高。


基于`got.js`封装一个返回流对象的异步请求方法：


```typescript
import { Injectable } from '@nestjs/common'
import * as got from 'got'
import KeepAliveAgent from 'agentkeepalive'

const keepaliveAgent = new KeepAliveAgent({
  maxSockets: 1000,
  timeout: 60000,
  freeSocketTimeout: 15000,
  socketActiveTTL: 60000
}); // 长连接配置

@Injectable()
export class HttpClientService {
  constructor(
    private config: ConfigService,
    private logger: MyLogger,
    private requestTracking: RequestTracking
  ) { }

  public async streamRequest(
    url: string,
    gotConfig?
  ): Promise<{
    statusCode: number;
    headers: { [key: string]: string };
    stream: any;
  }> {
    const baseGotConfig = {
      responseType: 'buffer',
      retry: 0,
      followRedirect: false,
      methodRewriting: false,
      decompress: false,
      timeout: 50 * 1000,
      throwHttpErrors: false,
      isStream: true,
      agent: { http: keepaliveAgent },
      ...gotConfig
    }
    const res = await got(url, gotConfig);

    return new Promise((resolve, reject) => {
      res.on('response', (response) => {
        resolve({
          statusCode: response.statusCode,
          headers: response.headers,
          stream: res, // res上的stream属性是一个流对象
        });
      });
      res.on('error', error => reject(error));
    });
  }
}
```


NestJS底层是Express，Express请求上的Response是一个流对象，可以直接调用其上的`write()`方法向返回流中写入数据。因此，这里就是将`streamRequest()`返回的流对象中的数据，传输（pipe）到返回流中。


在Controller中处理逻辑的伪代码是：


```typescript
import { Controller } from '@nestjs/common';
import * as stream from 'stream';
import * as util from 'util';

const pipeline = util.promisify(stream.pipeline);

@Controller()
export class AppController {
  constructor(
    private readonly httpClient: HttpClientService,
    private readonly errorFactory: ErrorFactory
  ) { }

  async dockerHandler(req, res) {
    try {
      const streamRes = await this.httpClient.streamRequest('xxxxx', {})
      // ...
      await pipeline(streamRes.stream, res);
      // ...
    } catch (e) {
      throw this.errorFactory.StreamRequestError({
        message: e.message,
        stack: e.stack
      });
    }
  }
}

```


## 鉴权模块


> 鉴权模块的作用是什么？  
> 防止用户资源被越权访问。


在C/S架构中，身份认证的需求很常见。根据不同的场景，有以下几种解决方案：

- 基于Cookie：客户端需要保存数据，数据类型和大小受限，并且浪费宽带
- 基于Session：服务端需要保存sessionId，服务端是由状态的，不方便快速横向扩展，不适合分布式服务场景
- 基于Token：
	- 和session相比：无需在客户端和服务端保存多余状态，无状态服务器能够快速地进行横向扩展
	- 和cookie相比：请求中只需要携带token即可，不需要携带多余cookie

由于网关是无状态服务，因此选用了基于Token的方案。


### JSON Web Token


![008i3skNgy1gsis2b82toj30oe08cq2y.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsis2b82toj30oe08cq2y.jpg)


基于Token最常用的规范是JWT。JWT由 3 部分组成：`header.payload.sign`。


**header部分**：记录metadata的json对象。


```typescript
{
    "alg": "hs256", // 加密算法
    "typ": "JWT" // 签名类型：就是jwt
}

```


**payload部分**：记录不敏感数据的json对象。


```json
{
    "userName": "dongyuanxin"
}
```


**sign部分**：将header和payload组成的数据，通过非对称加密算法，得到的一串签名。伪代码如下：


```typescript
const data = base64url.encode(header) + "." + base64url.encode(payload);
const sign = hash.hs256(secret, data).toString(); // 得到签名
```


### TCB 用户体系鉴权流程


![008i3skNgy1gsis2ryg21j30p20no3yr.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsis2ryg21j30p20no3yr.jpg)


如果用户在云开发控制台中，开启了HTTP访问服务的某个路由的鉴权，那么当请求抵达Gateway的时候，Gateway会使用密钥，校验Token的有效性。


整体流程是：

- 客户端APP调用tcb-js-sdk，使用账密登录TCB用户体系
- 数据流服务（图中的认证服务）会校验账密是否正确，正确的情况下，使用密钥签出JWT
- 客户端APP获取JWT，并且将其放入在HTTP Header中，请求Gateway
- Gateway使用密钥校验JWT有效性，如果签名有效并且未过期，那么返回数据

## 缓存模块


![008i3skNgy1gsird3n7gnj318z0u076r.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2021-07-16-tcb-gateway/008i3skNgy1gsird3n7gnj318z0u076r.jpg)


对于网关系统来说，大量数据都来自底层服务的接口，是个典型的「读密集」服务。对于读密集服务来说，缓存是提高系统性能、降低接口响应时间的大杀器。


在没有上缓存之前，请求经过网关的耗时曾经高达50ms。这对于一个网关系统来说，无法接受。主要原因是：

- 网关调用了大量的底层服务接口，作为执行鉴权、转发等逻辑的依据。这里自然就形成了瓶颈。
- 由于没有缓存，每次请求都要对底层服务接口发起请求，底层服务负载高

为了解决这个问题，网关系统中，加入了本地缓存（Local Cahe）、中心化缓存（Redis Cache），并且还设计了缓存层以及异步更新机制。


### 本地缓存（Local Cache）


又名内存缓存。将Cache Value放入Node内存中，如果命中了Cache Key，则直接读取，性能非常高。


但是随着数据增多，需要定期对缓存进行清除，否则会造成Node内存被吃满；同时，缓存热点数据、最近访问数据，比单纯地缓存普通数据对平均性能的提升更明显。


基于这些考虑，本地缓存基于「缓存淘汰算法」（LRU算法）实现，并且为缓存加入了过期时间属性。缓存的数据结构如下：


```typescript
import QuickLru from '@/utils/quick-lru';

type LruCache = QuickLru<string, LruCache>

interface LruCacheValue {
    value: any; // 缓存值
    expire: number; // 过期时间
}
```


本地缓存的读写流程是：

- 读：查询key是否存
	- 不存在，返回空
	- 存在：检查是否过期，过期的话，清除过期值，返回空；否则，返回缓存值
- 写：写入内存，设置过期时间

### 中心化缓存（Redis Cache）


本地缓存存在什么问题呢？网关服务有多个地域，每个地域有多个可用区。在网关服务之前有一层Ingress（见前面的图），由于网关是无状态的，因此Ingress配置了性能最高的「无加权轮询算法」来保证负载均衡。


假设集群有 N 个 Pod，并且 N 远大于 60，那么对于同一个用户，当 1min 请求不到 60 次，请求就会均匀落到各个 Pod，那么堆上缓存其实是无用的。**这就导致了缓存命中率在 40%左右，链路耗时也较高（因为缓存没共享，每次都要重新计算）。**


为了提高缓存命中率，这里就需要一个中心化的缓存存储，给多个网关服务调用。这里中心化缓存系统就是Redis。


在Redis中，可以设置缓存过期时间（单位为秒）：`EXPIRE keyName seconds` 。过期之后，不需要客户端手动删除，缓存会被自动删除。


### 多层缓存（Layer Cache）


可以将本地缓存和中心化缓存理解成2个层。


多层缓存的读顺序是：

1. 读取本地缓存，命中后判断过期时间，成功则返回；否则进入下一步
2. 读取中心化缓存，命中则返回，否则返回空

多层缓存的写顺序是：

1. 更新本地缓存
2. 更新中心缓存

### 异步刷新策略（Async Flush）


异步更新策略采取的是「被动模式」刷新缓存。为了支持异步更新，需要修改下前面定义的缓存数据结构，新增`deleteTime`字段：


```typescript
interface LruCacheValue {
    value: any; // 缓存值
    expireTime: number; // 失效时间（此期间可刷新）
    deleteTime: number; // 删除时间（超过此时间，需要重新请求数据）
}
```


新旧缓存结构对比：

- 旧：时间超过expireTime，缓存会自动失效，并且删除
- 新：时间超过expireTime后，缓存会失效，但不会删除。
	- 在[expireTime, deleteTime] 这段时间：如果有请求命中缓存，那么会先返回老缓存（保证接口响应）数据，再异步地去请求接口，更新缓存
	- 超过deleteTime之后，缓存失效，并且会删除

这么说可能还是有点抽象，来看下本文的最后一段伪代码（支持异步刷新策略的多层缓存）：


```typescript
import { Injectable, Scope } from '@nestjs/common';
import QuickLRU from 'quick-lru';

@Injectable({ scope: Scope.TRANSIENT })
export class CacheLayerService {
  private _cache: LruCache;
  private _ttl: number; // 缓存有效期，默认为1分钟

  constructor(ttl = 60 * 1000) {
    this._cache = new QuickLRU({ maxSize: 1000 });
    this._ttl = ttl;
  }

  /**
   * 读取缓存，缓存过期自动回源，回源成功则自动续期
   *
   * @param {any} key 缓存标识
   * @param {Function} fn 数据回源函数，返回一个 Promise 对象
   * @param {number} deleteTime 回源获得的缓存的最终过期时间，默认为 Infinity
   * @param {boolean} isAsync 是否异步回源，默认异步
   */
  public async getWithBack(key, fn: IFunction<any>, deleteTime?: number, isAsync = true) {
    const data = this._cache.get(key);
    if (!data) {
      return;
    }

    const now = Date.now();
    // 情况1: 缓存未过期
    if (now <= data.expireTime) {
      return data.value;
    }
    // 情况2: 缓存过期，并且超过了最大过期时间
    if (now > data.deleteTime) {
      this._cache.delete(key);
      return;
    }

    // 情况3: 缓存过期，但是没有超过最大过期时间
    if (isAsync) {
      // 异步回源续期
      fn()
        .then(value => this.set(key, value, deleteTime))
        .catch(error => {
          // ignore error
        });

      return data.value;
    } else {
      // 同步回源续期
      try {
        const value = await fn();
        this.set(key, value, deleteTime);
        return value;
      } catch (error) {
        // ignore error
        return data.value;
      }
    }
  }
}
```


## 总结


现代系统架构中，在前端和后端之间通常有一层网关（也叫中间层、应用层），来承担限流、缓存、鉴权等相对独立的逻辑。在日常开发中，Nginx就是耳熟能详的经典网关。


**而网关作为承接前端所有流量的入口，服务需要保证高可用和高性能**。文中提到的日志模块、监控告警模块是为了保证服务可用性，DNS模块、流量转发、缓存模块是为了提高服务性能。除此之外，网关系统还做了多地域多可用区部署、灰度/普通/VIP多集群部署，从而最大程度地保证服务的可用性。这里由于篇幅原因不再展开。


**同时，服务在设计上也需要考虑可扩展性**。对于本文提到的系统，可扩展性主要提现在代码设计，比如：网关内部模块的分层架构设计，鉴权模块中的鉴权npm库等。


希望这篇文章可以为你设计、开发和部署运维大型Node服务提供一些思路。


> 文章原文来自：编程理想国-亿级流量·云开发网关架构设计，转载请标明出处。


	**声明：文中不涉及任何内部截图和敏感信息，均使用公开的产品截图，以及与业务无实际关联的抽象模型。仅做技术交流和分享**。


