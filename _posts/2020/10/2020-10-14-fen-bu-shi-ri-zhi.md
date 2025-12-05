---
title: "分布式日志设计"
url: "2020-10-14-fen-bu-shi-ri-zhi"
date: 2020-10-14
---

国庆前后腾讯云·云开发开发了分布式全链路日志。相较于单体服务日志，分布式全链路日志的设计更有意思，包括日志生成逻辑、上报逻辑以及前端的显示逻辑。


## 全链路日志的作用


用户从发起请求，到收到请求之间，请求会经过很多服务。


例如对于云开发来说，用户通过 js-sdk 调用云数据库，写入数据。链路是：js-sdk => LB（负载均衡）=> gateway（云开发中间层） => backend => db。


分布式链路日志就可以看到整条链路上的，每次 rpc 调用，产生的日志。并且可以将其串联起来。


除了收归串联起整体服务日志，还可以作为依据，优化链路节点耗时、发现调用量大的服务。


## 全链路日志字段


child_database


## 全链路日志上报


假设 rpc 调用场景如下：


![007S8ZIlgy1gjorxpsadwj30i205qdg7.jpg](https://raw.githubusercontent.com/dongyuanxin/static/main/blog/imgs/2020-10-14-fen-bu-shi-ri-zhi/007S8ZIlgy1gjorxpsadwj30i205qdg7.jpg)


### 简单调用上报


只涉及一次 rpc 调用，被调方不发起新的 rpc，直接返回结果。


### 调用方


step1：针对整条链路，生成 traceid，作为链路标识。


> 注意：只有首跳节点，才生成 traceid，其他节点通过 headers、环境变量等方式读取前面节点传来的 traceid。


step2：cgi 作为第一跳，没有父 rpc，所以 childof 字段为空


step3：向 svr1 发起 rpc 调用前，生成 spanid、start_time 等字段


step4: 发送 rpc 调用，并且向 svr1 传递 spanid、traceid（通过 headers、环境变量等方式）


step5：收到 rpc 调用返回，计算 cost_time


step6：组装字段，向日志平台/后端上报日志


### 被调方


step1: svr1 收到 cgi 的调用


step2: （通过 headers、环境变量等方式）读取到 spanid、traceid。


step3：直接处理逻辑，不涉及其他 rpc 调用，将结果返回


### 复杂调用上报


一次 rpc 调用涉及多次 rpc 调用，被调方需要通过多个 rpc 来组装结果


基本逻辑和上一部分一致。要注意的是被调方发向 svr2 和 svr3 各发起一次 rpc 调用：

- 两次 rpc 的 childof 字段就是 cgi 传给 svr1 的 spanid 字段
- 两次 rpc 的 spanid 都是新生成，然后向下个节点传递

## 参考

- [调用链 trace 的设计分析](https://yuerblog.cc/2017/06/22/talk-about-rpc-trace/)
- [天机阁——全链路跟踪系统设计与实现](https://www.infoq.cn/article/JF-144XPDqDxxdizdfwT)：上报逻辑有所区别，一次 rpc 调用方和被调方都上报，两个子 span 再合并成一个完整 span。

