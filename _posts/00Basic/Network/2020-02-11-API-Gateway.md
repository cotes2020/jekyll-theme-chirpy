---
title: AWS - VPC Gateway - API Gateway
date: 2020-02-11 11:11:11 -0400
categories: [00Basic, Network]
tags: [Basic, AWS, Network]
toc: true
image:
---

- [API Gateway](#api-gateway)
  - [background](#background)
  - [basic](#basic)
  - [安全性（身份验证和潜在的授权）](#安全性身份验证和潜在的授权)
  - [开源 APIGateway](#开源-apigateway)
  - [单节点 API 网关](#单节点-api-网关)
  - [Backends for frontends 网关](#backends-for-frontends-网关)
- [AWS API Gateway](#aws-api-gateway)
- [Ali API Gateway](#ali-api-gateway)
  - [basic](#basic-1)
  - [use case](#use-case)
    - [API Gateway + Function Compute](#api-gateway--function-compute)

---

# API Gateway

> ProgrammableWeb，该数据库自 2005 年以来一直在收集开放的 API。
> 2005 年仅列出了大约 100 种 API，如今已有超过 10,000 个公共 API
> 这种增长越来越依赖于用户数据资料库的经济, Salesforce 通过 API 创造了其 30 亿美元年收入的 50％以上，以及 Expedia 20 亿美元年收入的近 90％。
> 公司通过以各种方式计算对 API 及其背后资源的访问来获得 API 收入。
> 例如，Twitter，Facebook 和其他提供基于广告的 API，这些 API 允许基于报告和分析来进行有针对性的广告，但是广告代理商和其他品牌必须为访问这些 API 付费。



## background

从单一应用程序切换到微服务时，客户端的行为不能与客户端具有该应用程序的一个入口点的行为相同。
- 微服务上的某一部分功能与单独实现该应用程序时存在不同。
- 客户端必须处理**微服务体系结构**带来的所有复杂性，例如
  - 聚合来自各种服务的数据，维护多个端点，客户端和服务器之间的联系增加
  - 对每个服务进行单独的身份验证等，
  - 同时客户端对微服务的依赖性也直接导致了重构服务的困难。
- 一种直观的方法是将这些服务隐藏在新的服务层后面，并提供针对每个客户端量身定制的 API。
  - 来自客户端的所有请求都首先通过 API 网关，然后网关再将请求转到适当的微服务。
- 该聚合器服务层: API 网关，它是解决此问题的常用方法。

微服务架构风格
- 一个大应用被拆分成为了多个小的服务系统提供出来，这些小的系统他们可以自成体系，也就是说这些小系统可以拥有自己的数据库，框架甚至语言等，这些小系统通常以提供 `Rest Api` 风格的接口来被 H5, Android, IOS 以及第三方应用程序调用。
- 但是在UI上进行展示的时候，我们通常需要在一个界面上展示很多数据，这些数据可能来自于不同的微服务中


例子。
- 电商系统中，查看一个商品详情页，这个商品详情页包含商品的标题，价格，库存，评论等，
- 这些数据对于后端来说可能是位于不同的微服务系统之中，
- 后台的系统可能是这样来拆分服务的：
  - 产品服务 - 负责提供商品的标题，描述，规格等。
  - 价格服务 - 负责对产品进行定价，价格策略计算，促销价等。
  - 库存服务 - 负责产品库存。
  - 评价服务 - 负责用户对商品的评论，回复等。
- 现在，商品详情页需要从这些微服务中拉取相应的信息，
- 问题
  - 微服务系统架构没办法像传统单体应用一样依靠数据库的 join 查询来得到最终结果，那么如何才能访问各个服务呢？
  - 按照微服务设计的指导原则，我们的微服务可能存在下面的问题：
    - 服务使用了多种协议，因为不同的协议有不同的应场景用，比如可能同时使用 HTTP, AMQP, gRPC 等。
    - 服务的划分可能随着时间而变化。
    - 服务的实例或者Host+端口可能会动态的变化。
  - 那么，对于前端的UI需求也可能会有以下几种：
    - 粗粒度的API，而微服务通常提供的细粒度的API，对于UI来说如果要调用细粒度的api可能需要调用很多次，这是个不小的问题。
    - 不同的客户端设备可能需要不同的数据。Web,H5,APP
    - 不同设备的网络性能，对于多个api来说，这个访问需要转移的服务端会快得多
  - 以上，就是我们构建微服务的过程中可能会遇到的问题。
  - 解决: API 网关（API Gataway）。

![Screen Shot 2022-02-11 at 13.56.03](https://i.imgur.com/llsTrSt.png)


## basic

API网关方式的核心要点是，所有的客户端和消费端都通过统一的网关接入微服务，在网关层处理所有的非业务功能。
- 通常，网关也是提供REST/HTTP的访问API。
- 服务端通过APIGW注册和管理服务。


典型的 API 网关包括
- 安全性（身份验证和潜在的授权）
- 管理访问配额和限制
- 缓存（代理语句和缓存）
- API 的组成和处理
- 路由（“中转器”）到“内部” API
- API 运行状况监视（性能监视）
- 版本控制（自动化流程）

API 网关的优势
- 在统一的位置管理和实施
- 将大部分问题外部化，因此简化了 API 源代码
- 提供 API 的管理中心和视图，更方便采用一致的策略

API 网关的缺点
- 容易出现单点故障或瓶颈
- 由于所有 API 规则都在一个位置，因此存在复杂性风险
- 被锁定的风险，日后系统迁移并不简单



## 安全性（身份验证和潜在的授权）

访问控制是 API 网关技术的第一大安全驱动程序
- 管理谁能访问 API 并建立有关如何处理数据请求的规则。
- 访问控制几乎能扩展到建立其他策略，包括对某些来源的 API 调用的速率限制，甚至是通过 API 访问所有或某些资源的要求。

API 网关的访问控制功能通常从身份验证机制开始，以确定任何 API 调用的实际来源。
- 当前，最流行的网关是 OAuth，它充当中介程序，用于访问基于 Web 的资源而不向服务公开密码，并且基于身份验证进行保留，在这种情况下企业可以承受丢失数据的麻烦，确保密钥完全保密。


![Screen Shot 2022-02-11 at 14.25.37](https://i.imgur.com/wOS3GWw.png)

- 通信安全
  - 网关是一种通过单个通道连接所有 API 服务以评估，转换和保护整个组织中通讯的好方法。
  - 当所有流量都通过网关进行转接时，IT 安全专家能够动态到所有的项目动态。
  - API 网关可以在内部服务之间引入消息安全性，从而使内部服务更加安全，并且在服务之间来回传递的消息经过加密。
  - 即便使用传输层加密（TLS），忽略正确的身份验证也会导致问题。
  - 例如，在 API 请求中使用有效的手机号码，任何人都可以获取个人电子邮件地址和设备标识数据。
  - 像 `OAuth / OpenIDConnect` 这样的行业标准强大的身份验证和授权机制，以及 TLS，都是至关重要的。

- 威胁防护
  - 没有威胁防护，API 网关，其 API 和集成服务器的本机服务基本上是不安全的。
  - 这意味着潜在的黑客，恶意软件或任何匿名的外部人员都可以轻松地尝试传播一系列攻击，例如 DDoS 或 SQL 注入。
  - API 是企业与世界进行数字连接的网关。不幸的是，有些恶意用户旨在通过注入“额外”的命令或表达式来删除，更新甚至创建可用于 API 的任意数据来访问后端系统。
  - 例如，2014 年 10 月，Drupal 宣布了一个 SQL 注入漏洞，该漏洞使攻击者可以访问数据库，代码和文件目录。甚至攻击最严重的程度是，攻击者可以将所有数据复制到客户端站点之外，这将对企业造成多大的影响。
  - 在现实中并不少见，我们已经不止一次地看到 API 在没有威胁防护的情况下上线了。
  - 注入威胁的类型有很多，
    - 最常见的是 SQL 注入、RegExInjection 和 XML 注入。
    - SQL 注入
      - SQL 注入保护使你可以阻止可能导致 SQL 注入攻击的请求。
    - JSON 威胁防护
      - JavaScript 对象表示法（JSON）容易受到内容级别的攻击。
      - 此类攻击试图使用巨大的 JSON 文件淹没解析器，并最终使服务崩溃。
    - XML 威胁防护
      - 对 XML 应用程序的恶意攻击通常涉及较大的递归有效负载，XPath / XSLT 或 SQL 注入，以及 CData，以淹没解析器并最终使服务崩溃。


- 信息保护
  - 许多 API 开发人员都习惯使用 200 代表成功请求，404 代表所有失败，500 代表内部服务器错误，在某些极端情况下，在详细的堆栈跟踪之上使用 200 代表带有失败消息的主体。
  - 当堆栈跟踪以程序包名称，类名称，框架名称，版本，服务器名称和 SQL 查询的形式揭示底层设计或体系结构实现时，可能会向恶意用户泄漏信息。
  - 合适的做法是返回一个“平衡”的错误对象，该对象具有正确的 HTTP 状态代码，所需的最少错误消息，并且在错误情况下不进行堆栈跟踪。
  - 这将改善错误处理并保护 API 实施细节免受攻击者的侵害。
  - API 网关可用于将后端错误消息转换为标准化消息，从而使所有错误消息看起来都标准化，这也消除了公开后端代码结构的麻烦和危险。

- 白名单和允许白名单的方法
  - 考虑 IP 地址级别的 API 流量，应该有设备，服务器，网络和客户端 IP 地址的已知列表。根据网络的紧密程度，此列表的大小会有所不同。
  - RESTful 服务很常见，它允许多种方法访问该实体上不同操作的给定 URL。
  - 例如，GET 请求可能会读取实体，而 PUT 将更新现有实体，POST 将创建新实体，而 DELETE 将删除现有实体。
  - 对于服务来说，适当地限制允许动词很重要，这样只有允许的动词请求才能起作用，而其他所有动词都将返回正确的响应码（例如，403 Forbidden）。


- 讯息大小
  - 有消息大小限制是很好的。如果你十分确认知道不会接收大文件消息（例如，超过 2MB），那限制大小过滤掉大文件消息能尽可能避免一些未知攻击。

- 限速
  - 需要对所有 API 用户进行身份验证，并记录所有 API 调用，从而使 API 提供程序可以限制所有 API 用户的使用率。
  - 许多 API 网关都允许你限制可以对任何单个 API 资源进行 API 调用的数量，以秒，分钟，天或其他相关约束条件来指定消耗量。


---


## 开源 APIGateway


通常情况下， API 网关要做很多工作，它作为一个系统的后端总入口，承载着所有服务的组合路由转换等工作，除此之外，我们一般也会把安全，限流，缓存，日志，监控，重试，熔断等放到 API 网关来做，那么可以试想在高并发的情况下，这里可能会出现一个性能瓶颈。

如果没有开源项目的支撑前提下，自己来做这样一套东西，是非常大的一个工作量，而且还要做 API 网关本身的高可用等，如果一旦做不好，有可能最先挂掉的不是你的其他服务，而就是这个API网关。

API 网关：开源

- Tyk：
  - Tyk是一个开放源码的API网关，它是快速、可扩展和现代的。
  - Tyk提供了一个API管理平台，其中包括API网关、API分析、开发人员门户和API管理面板。
  - Try 是一个基于Go实现的网关服务。
- Kong：
  - Kong是一个可扩展的开放源码API Layer(也称为API网关或API中间件)。
  - Kong 在任何RESTful API的前面运行，通过插件扩展，它提供了超越核心平台的额外功能和服务。
- Orange：
  - 和Kong类似也是基于OpenResty的一个API网关程序，是由国人开发的，学姐也是贡献者之一。
- Netflix zuul：
  - Zuul是一种提供动态路由、监视、弹性、安全性等功能的边缘服务。
  - Zuul是Netflix出品的一个基于JVM路由和服务端的负载均衡器。
- apiaxle:
  - Nodejs 实现的一个 API 网关。
- api-umbrella:
  - Ruby 实现的一个 API 网关。



![Screen Shot 2022-02-11 at 21.02.56](https://i.imgur.com/6sGNnOC.png)


**OpenResty Api Gateway**
- 从左至右 HTTP 请求先由DNS在拿到第一手流量后负载均衡到基于 OpenResty 的 API Gataway 网关集群，在这个流程我们可以使用像 Kong,Orage,Tyk 这些开源的支持高并发高访问量 API 网关程序在做第一层`流量的防护`
- 在这一级我们可以做一些像身份认证，安全，监控，日志，流控等策略。
- 除了这些我们还可以做一些服务的发现和注册（这个要看不同网关的支持程度），接口的版本控制，路由重写等。


**Aggr Api Gateway**
- 然后再由这些 API 网关把请求再负载到不同的 Aggr Api Gateway
- 在这里我们做聚合服务 这个操作，具体体现也就是图中的黄色区域是需要由各个语言的开发人员来需要写代码实现的。
- 具体流程也就是我们可以引入像 Ocelot 这种和语言相关的 API 网关开源项目，然后通过 NuGet 包引入之后通过 Json配置+聚合代码的方式来整合后端的各个微服务提供聚合查询等操作。
- 这期间对于有需求的接口，我们可以应用超时，缓存，熔断，重试等策略。
- 从 `Aggr Api Gateway` 到后`端微服务集群`这中间就属于内部的通讯了，我们可以使用对内部友好的通讯协议比如 gRPC 或者 AMQP 等，然后进行 RPC调用提高通讯性能。

注意 ：Aggr Api Gateway 这个网关对于一些接口来说的话并不是必须的，也可以由后端微服务直接提供REST API给第一层网关使用。

以上，就是我理解的 API 网关在整个微服务架构中的一个地位，承上启下，还是非常的重要。


---

## 单节点 API 网关

![Screen Shot 2022-02-11 at 20.54.08](https://i.imgur.com/IpqFoIr.png)

单节点的 API网关为每个客户端提供不同的API，而不是提供一种万能风格的API。
这个网关和微软在 eShop 项目中推荐的网关是一致的。


---


## Backends for frontends 网关

![Screen Shot 2022-02-11 at 20.55.32](https://i.imgur.com/WjkEPSE.png)

这种模式是针对不同的客户端来实现一个不同的API网关。



---

# AWS API Gateway


[link to detail page](https://ocholuo.github.io/posts/Gateway-IGW/)

---


# Ali API Gateway


## basic

- fully functional API hosting service
- publish the APIs centrally and offer them to the clients and partners. release the APIs on the Alibaba Cloud API marketplace where other developers may purchase them.

- a full API lifecycle management service. It provides `API definition and publishing, testing, release, and removal`.
- It can generate SDKs and API instructions and has a visualized debugging tool.
- increases the efficiency of API management and iterative release.
- flexibly and securely share their technological innovations leaving them free to focus on strengthening and advancing their businesses.

- provides multiple security and efficiency options that include `attack defenses, anti-replay, request encryption, identity authentication, permission management, and throttling`.
- has convenient maintenance, observation, and measurement tools, such as monitoring, alarms, and analysis.
- It also enables speedy and reliable microservice integration, front and back end separation, and robust system integration at low cost.


---

## use case

---

### API Gateway + Function Compute

- use Function Compute to create a function that returns a result to the API Gateway service
- monitor Alibaba Cloud API Gateway service directly from the Function Compute console.


**API Gateway Service**
1. Create an API in the API Gateway Service
   1. API Gateway > API Groups > Create API.
   2. Give the API a name and select the API Group.
   3. choose `No Certification` and leave the selections access restrictions empty
   4. ![pic](https://miro.medium.com/max/1200/0*ja4tOqnEnpBpOZRZ.png)
   5. Configure the API to have a `COMMON` Request Type using HTTP and HTTPS Protocols.
      1. `subdomain`: the **URL variable** required by the Function Compute function, make a note.
         1. **full path variable** for Function Compute function is: `https://..subdomain../fcpath/`
         2. Function Compute currently does not support **Match All Child Paths**.
      2. **request path** must contain the **Parameter Path** in the **request parameter** within brackets `[]`, For example: `/fcpath/[fcparam]`
   6. HTTP Method: `POST`
   7. Request Mode: `Request Parameter Passthrough`.
   8. ![pic](https://miro.medium.com/max/1200/0*SiQLeBx7VZ4YtzSA.png)

2. Define the path input parameter in the **Input Parameter Definition**.
   1. ![pic](https://miro.medium.com/max/1200/0*XB-hEDqtx20fsCuX.png)

3. Basic Backend Definition
   1. choose Function Compute as the Backend Service Type and select the correct Region.
   2. In the Service and Function name, add the Function Compute Service and Function names.
   3. For Role Arn, click Get Authorization.
   4. ![pic](https://miro.medium.com/max/1200/0*gKi9mUk5StYuLfPa.png)
   5. API Gateway service will automatically populate the field with the correct role details.
   6. ![pic](https://miro.medium.com/max/1200/0*LG-yYcrF07RsKUzC.png)

4. Backend Service Parameter Configuration, you have the details of the path parameter.
   1. ![pic](https://miro.medium.com/max/1200/0*xHv8XIsmI2wWRFue.png)

5. define the response in the final tab.
   1. Scroll down and click Create.
   2. ![pic](https://miro.medium.com/max/1200/0*rvJi1yow-H8YKjEg.png)
   3. You should see a successful result. Click OK.

6. Select the API in the list and click Deploy.
   1. Enter the deployment details and click OK.
   2. API details > Debug API.
   3. Add the HTTP protocol.
   4. Input and a header to reflect `Content-Type = application/json`
   5. Leave the certification as No Certificate.
   6. click Send Request.
   7. ![pic](https://miro.medium.com/max/798/0*L14J9fvL02ZiIx6P.png)
   8. You should see a 200 request success code in the output.




**Function Compute**
1. Create a **Function Compute** API Gateway Trigger Function
   1. Function Compute > add button
   2. give the service a name and slide open Advanced Settings and scroll down.
   3. ![pic](https://miro.medium.com/max/1200/0*1CcywlLtZc8Yi8mO.png)
   4. In Network config, allow Internet access
   5. In Role Config, configure the `AliyunApiGatewayFullAccess` role > Confirm Authorization Policy.

2. click through to the Service, create a Function.
   1. Click the add button next to Functions.
   2. On the template page, select the Empty Function.
   3. Our API Gateway function does not need a `Function Compute Trigger` so leave the No Trigger setting and click next in `Configure Triggers` tab.
   4. Give the Function a name and select the runtime environment > Python.
   5. ![pic](https://miro.medium.com/max/1200/0*6GQv-2IK9TLt4Uow.png)
   6. Check the details and configure System Policies for accessing the API Gateway cloud resource.
   7. Click Authorize > Click Confirm Authorization Policy, then Next > Create.

3. code Function Compute function.
   1. enter the following code into the code section.

    ```py
    # -*- coding: utf-8 -*-
    import json
    def handler(event, context):
        event = json.loads(event)
        content = {
            'path': event['path'],
            'method': event['httpMethod'],
            'headers': event['headers'],
            'queryParameters': event['queryParameters'],
            'pathParameters': event['pathParameters'],
        }

        rep = {
            "isBase64Encoded": "false",
            "statusCode": "200",
            "headers": {
                "x-custom-header": "no"
            },
            "body": content
        }
        return json.dumps(rep)
    ```

4. Click Event
   1. configure the Event parameters.
   2. ![pic](https://miro.medium.com/max/1200/0*qbEknyv6XgTC2UdQ.png)
   3. Add the Custom Event parameters and click OK.
   4. ![pic](https://miro.medium.com/max/1200/0*PvOtw7i9M1Frjdhe.png)
   5. Click Save and Invoke.
   6. should see a Success message.
   7. ![pic](https://miro.medium.com/max/1200/0*rt5gtUFK89s1Svkq.png)




**test the API Gateway**.
1. API Gateway API
   1. debug API page.
   2. The API Gateway is functional.
   3. ![pic](https://miro.medium.com/max/1200/0*L1UaPUb_Yy9mt7Mh.png)


**Monitoring API Gateway with Function Compute**
1. Function Compute
   1. click Monitoring
   2. a Monitoring Service Overview of Function Compute service usage over time.
   3. Service List tab > API Gateway Service
   4. a list of all the Functions running as part of the service. Click on a function to see more details.
   5. ![pic](https://miro.medium.com/max/1200/0*guNAYJdYNJ0rWyxA.png)
   6. the Function Monitoring Overview page, you will see detailed measurements over time for Total Invocations, their Average Duration, Function Errors, and Maximum Memory Usage for the function.
   7. Create Alarm Rule
      1. have the option to set an Alarm that will raise a warning whenever the function is under stress, or load, or erroring on a number of different monitoring parameters.
   8. Any Alarm Rules you set are listed under the Alarm Rule tab.

Reference:
- [https://www.alibabacloud.com/blog/using-api-gateway-with-alibaba-clouds-function-compute\_594695?spm=a2c41.12784890.0.0](https://www.alibabacloud.com/blog/using-api-gateway-with-alibaba-clouds-function-compute_594695?spm=a2c41.12784890.0.0)
